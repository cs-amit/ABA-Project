package com.aba.smartsleep.app.data

import androidx.room.withTransaction
import com.aba.smartsleep.app.transport.PhoneBatchDelivery
import com.aba.smartsleep.app.transport.PhoneBatchInbox
import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeaturePipeline
import com.aba.smartsleep.core.features.FeaturePipelineState
import com.aba.smartsleep.core.model.AlarmSettings
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.model.SessionStatus
import com.aba.smartsleep.core.model.SleepSession
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

interface SessionRepository {
    suspend fun append(delivery: PhoneBatchDelivery): Boolean
    fun session(sessionId: String): Flow<SleepSession?>
}

class RoomSessionRepository(
    private val database: AppDatabase,
    private val nowEpochMillis: () -> Long = System::currentTimeMillis,
    private val beforeDurableCommit: suspend () -> Unit = {},
) : SessionRepository {
    private val transactionMutex = Mutex()
    private val pipelines = mutableMapOf<String, FeaturePipeline>()

    override suspend fun append(delivery: PhoneBatchDelivery): Boolean = transactionMutex.withLock {
        var committedPipeline: FeaturePipeline? = null
        database.withTransaction {
            val inserted = database.persistedBatchDao().insertIgnore(
                PersistedPhoneBatchEntity(delivery.key.sessionId, delivery.key.sequence, nowEpochMillis()),
            )
            if (inserted != -1L) {
                val current = pipelines[delivery.batch.sessionId] ?: loadPipeline(delivery.batch.sessionId)
                val staged = FeaturePipeline.fromState(current.snapshot())
                val epochs = staged.consume(delivery.batch.samples)
                beforeDurableCommit()
                database.featureEpochDao().insertAll(epochs.map { it.toEntity() })
                database.featurePipelineCheckpointDao().upsert(staged.snapshot().toEntity())
                database.watchAcknowledgementDao().insertIgnore(
                    PendingWatchAcknowledgementEntity(delivery.key.sessionId, delivery.key.sequence),
                )
                committedPipeline = staged
            }
        }
        committedPipeline?.let { pipelines[delivery.batch.sessionId] = it }
        // The transaction and checkpoint are durable before the release journal can be completed.
        delivery.confirmPersisted()
    }

    suspend fun pendingWatchAcknowledgements(): List<PhoneBatchInbox.DeliveryKey> =
        database.watchAcknowledgementDao().pending().map { PhoneBatchInbox.DeliveryKey(it.sessionId, it.sequence) }

    suspend fun markWatchAcknowledged(key: PhoneBatchInbox.DeliveryKey) {
        database.watchAcknowledgementDao().delete(key.sessionId, key.sequence)
    }

    suspend fun recordPrediction(event: PredictionEventEntity) = database.predictionEventDao().insert(event)
    suspend fun recordAlarm(event: AlarmEventEntity) = database.alarmEventDao().insert(event)
    suspend fun recordRefreshedFeeling(feedback: RefreshedFeelingFeedbackEntity) = database.refreshedFeelingFeedbackDao().insert(feedback)

    suspend fun upsertSession(session: SleepSession, researchRetentionEnabled: Boolean = false) {
        database.withTransaction {
            database.sleepSessionDao().upsert(session.toEntity(researchRetentionEnabled))
            database.alarmSettingsDao().upsert(session.settings.toEntity(session.sessionId))
        }
    }

    override fun session(sessionId: String): Flow<SleepSession?> =
        database.sleepSessionDao().observe(sessionId).map { entity -> entity?.toModel() }

    private suspend fun loadPipeline(sessionId: String): FeaturePipeline {
        val checkpoint = database.featurePipelineCheckpointDao().checkpoint(sessionId) ?: return FeaturePipeline(sessionId)
        return FeaturePipeline.fromState(checkpoint.toState())
    }

    private fun FeatureEpoch.toEntity() = FeatureEpochEntity(sessionId, startEpochMillis, endEpochMillis, values, validForInference)

    private fun FeaturePipelineState.toEntity() = FeaturePipelineCheckpointEntity(
        sessionId = sessionId,
        currentStartEpochMillis = currentStartEpochMillis,
        lastTimestampEpochMillis = lastTimestampEpochMillis,
        pendingSamplesPayload = encodePendingSamples(pendingSamples),
        baselinePayload = encodeBaseline(priorValidRawValues),
    )

    private fun FeaturePipelineCheckpointEntity.toState(): FeaturePipelineState {
        return FeaturePipelineState(sessionId, currentStartEpochMillis, lastTimestampEpochMillis, decodePendingSamples(pendingSamplesPayload), decodeBaseline(baselinePayload))
    }

    /** Local Room checkpoint format deliberately has no Data Layer payload cap. */
    private fun encodePendingSamples(samples: List<SensorSample>): ByteArray = ByteArrayOutputStream().use { bytes ->
        DataOutputStream(bytes).use { output ->
            output.writeInt(samples.size)
            samples.forEach { sample ->
                output.writeLong(sample.timestampEpochMillis)
                output.writeNullableFloat(sample.accelX); output.writeNullableFloat(sample.accelY); output.writeNullableFloat(sample.accelZ)
                output.writeNullableFloat(sample.heartRateBpm)
                output.writeBoolean(sample.ibiMillis != null); sample.ibiMillis?.let(output::writeInt)
                output.writeInt(sample.quality.ordinal)
            }
        }; bytes.toByteArray()
    }

    private fun decodePendingSamples(payload: ByteArray): List<SensorSample> = DataInputStream(ByteArrayInputStream(payload)).use { input ->
        List(input.readInt()) {
            SensorSample(input.readLong(), input.readNullableFloat(), input.readNullableFloat(), input.readNullableFloat(), input.readNullableFloat(), if (input.readBoolean()) input.readInt() else null, SensorQuality.entries[input.readInt()])
        }.also { require(input.available() == 0) }
    }

    private fun DataOutputStream.writeNullableFloat(value: Float?) { writeBoolean(value != null); value?.let(::writeFloat) }
    private fun DataInputStream.readNullableFloat(): Float? = if (readBoolean()) readFloat() else null

    private fun encodeBaseline(values: List<FloatArray>): ByteArray = ByteArrayOutputStream().use { bytes ->
        DataOutputStream(bytes).use { output ->
            output.writeInt(values.size)
            values.forEach { row ->
                output.writeInt(row.size)
                row.forEach(output::writeFloat)
            }
        }
        bytes.toByteArray()
    }

    private fun decodeBaseline(payload: ByteArray): List<FloatArray> = DataInputStream(ByteArrayInputStream(payload)).use { input ->
        List(input.readInt()) {
            FloatArray(input.readInt()) { input.readFloat() }
        }.also { require(input.available() == 0) { "Checkpoint baseline has trailing bytes." } }
    }
}

private fun SleepSession.toEntity(researchRetentionEnabled: Boolean) = SleepSessionEntity(sessionId, startedAtEpochMillis, endedAtEpochMillis, status.name, researchRetentionEnabled)
private fun AlarmSettings.toEntity(sessionId: String) = AlarmSettingsEntity(sessionId, targetEpochMillis, wakeWindowMinutes, probabilityThreshold)
private fun SessionWithAlarmSettings.toModel(): SleepSession = SleepSession(
    sessionId = sessionId,
    settings = AlarmSettings(targetEpochMillis, wakeWindowMinutes, probabilityThreshold),
    startedAtEpochMillis = startedAtEpochMillis,
    status = SessionStatus.valueOf(status),
    endedAtEpochMillis = endedAtEpochMillis,
)
