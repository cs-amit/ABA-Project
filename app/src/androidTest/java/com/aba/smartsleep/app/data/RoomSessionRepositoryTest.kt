package com.aba.smartsleep.app.data

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.aba.smartsleep.app.transport.PhoneBatchDelivery
import com.aba.smartsleep.app.transport.PhoneBatchInbox
import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.transport.SensorBatchCodec
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RoomSessionRepositoryTest {
    private val database = Room.inMemoryDatabaseBuilder(
        InstrumentationRegistry.getInstrumentation().targetContext,
        AppDatabase::class.java,
    ).allowMainThreadQueries().build()
    private val repository = RoomSessionRepository(database)

    @After
    fun closeDatabase() = database.close()

    @Test
    fun duplicateDeliveryPersistsOneFeatureEpochAndConfirmsAfterTransaction() = runBlocking {
        var confirmations = 0
        val delivery = delivery(sequence = 7) { confirmations += 1; true }

        assertTrue(repository.append(delivery))
        assertTrue(repository.append(delivery))

        assertEquals(1, database.featureEpochDao().countForSession("session-1"))
        assertEquals(1, database.persistedBatchDao().countForSession("session-1"))
        assertEquals(2, confirmations)
    }

    @Test
    fun falseConfirmationDoesNotReportDeliveryCompleteBeforeDurableTransaction() = runBlocking {
        val delivery = delivery(sequence = 8) { false }

        assertFalse(repository.append(delivery))

        assertEquals(1, database.featureEpochDao().countForSession("session-1"))
        assertEquals(1, database.persistedBatchDao().countForSession("session-1"))
    }

    @Test
    fun partialWindowCheckpointSurvivesRepositoryRecreationBeforeDeliveryConfirmation() = runBlocking {
        assertTrue(repository.append(delivery(sequence = 10, samples = samples(endInclusive = 29_000)) { true }))
        val recreated = RoomSessionRepository(database)

        assertTrue(recreated.append(delivery(sequence = 11, samples = listOf(sampleAt(30_000))) { true }))

        assertEquals(1, database.featureEpochDao().countForSession("session-1"))
        assertEquals(1, database.featurePipelineCheckpointDao().countForSession("session-1"))
    }

    @Test
    fun failedTransactionDoesNotAdvancePipelineBeforeReplay() = runBlocking {
        var failCommit = true
        val repository = RoomSessionRepository(database, beforeDurableCommit = {
            if (failCommit) error("forced transaction failure")
        })
        val batch = delivery(sequence = 12, samples = samples()) { true }

        try {
            repository.append(batch)
            fail("Expected forced transaction failure")
        } catch (_: IllegalStateException) {
            // Expected: the Room transaction must roll back without advancing the active pipeline.
        }
        failCommit = false

        assertTrue(repository.append(batch))
        assertEquals(1, database.featureEpochDao().countForSession("session-1"))
    }

    @Test
    fun eventDaosPersistPredictionAlarmAndFeedbackRecords() = runBlocking {
        database.predictionEventDao().insert(PredictionEventEntity(sessionId = "session-1", occurredAtEpochMillis = 1, probability = .8f, validInput = true))
        database.alarmEventDao().insert(AlarmEventEntity(sessionId = "session-1", occurredAtEpochMillis = 2, reason = "light"))
        database.refreshedFeelingFeedbackDao().insert(RefreshedFeelingFeedbackEntity(sessionId = "session-1", rating = 4, recordedAtEpochMillis = 3))

        assertEquals(1, database.predictionEventDao().countForSession("session-1"))
        assertEquals(1, database.alarmEventDao().countForSession("session-1"))
        assertEquals(1, database.refreshedFeelingFeedbackDao().countForSession("session-1"))
    }

    @Test
    fun pendingWatchAcknowledgementIsDurableUntilMarkedDelivered() = runBlocking {
        assertTrue(repository.append(delivery(sequence = 13) { true }))
        val key = PhoneBatchInbox.DeliveryKey("session-1", 13)

        assertEquals(1, database.watchAcknowledgementDao().pending().size)
        repository.markWatchAcknowledged(key)
        assertEquals(0, database.watchAcknowledgementDao().pending().size)
    }

    @Test
    fun codecSplitBatchesUseUncappedLocalCheckpointAndReleaseTransformedSamples() = runBlocking {
        val denseSamples = (0L until 3_000L).map(::sampleAt)
        val first = SensorBatch("session-1", denseSamples.take(2_700))
        val second = SensorBatch("session-1", denseSamples.drop(2_700))
        assertTrue(SensorBatchCodec.encode(first).size < 90 * 1024)
        assertTrue(SensorBatchCodec.encode(second).size < 90 * 1024)
        try {
            SensorBatchCodec.encode(SensorBatch("session-1", denseSamples))
            fail("Combined transport payload should exceed the codec cap")
        } catch (_: IllegalArgumentException) {
            // The same rows still have to fit the Room-local checkpoint.
        }

        assertTrue(repository.append(delivery(sequence = 20, samples = first.samples) { true }))
        assertTrue(repository.append(delivery(sequence = 21, samples = second.samples) { true }))
        val openCheckpoint = database.featurePipelineCheckpointDao().checkpoint("session-1")!!
        assertTrue(openCheckpoint.pendingSamplesPayload.size > 90 * 1024)

        assertTrue(repository.append(delivery(sequence = 22, samples = listOf(sampleAt(30_000))) { true }))

        assertEquals(3, database.persistedBatchDao().countForSession("session-1"))
        assertEquals(1, database.featureEpochDao().countForSession("session-1"))
        val transformedCheckpoint = database.featurePipelineCheckpointDao().checkpoint("session-1")!!
        assertTrue(transformedCheckpoint.pendingSamplesPayload.size < 100)
    }

    private fun delivery(
        sequence: Long,
        samples: List<SensorSample> = samples(),
        confirmation: () -> Boolean,
    ): PhoneBatchDelivery = PhoneBatchDelivery(
        key = PhoneBatchInbox.DeliveryKey("session-1", sequence),
        batch = SensorBatch("session-1", samples),
        confirm = confirmation,
    )

    private fun samples(endInclusive: Long = 30_000): List<SensorSample> = (0L..endInclusive step 1_000L).map(::sampleAt)

    private fun sampleAt(timestamp: Long): SensorSample =
        SensorSample(
            timestampEpochMillis = timestamp,
            accelX = 1f,
            accelY = 0f,
            accelZ = 0f,
            heartRateBpm = 60f,
            ibiMillis = 1_000,
            quality = SensorQuality.VALID,
        )
}
