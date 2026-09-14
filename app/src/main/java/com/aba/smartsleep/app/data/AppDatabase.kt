package com.aba.smartsleep.app.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import kotlinx.coroutines.flow.Flow
import java.nio.ByteBuffer
import java.nio.ByteOrder

@Entity(tableName = "sleep_sessions")
data class SleepSessionEntity(
    @PrimaryKey val sessionId: String,
    val startedAtEpochMillis: Long,
    val endedAtEpochMillis: Long?,
    val status: String,
    val researchRetentionEnabled: Boolean = false,
)

@Entity(
    tableName = "alarm_settings",
    foreignKeys = [ForeignKey(
        entity = SleepSessionEntity::class,
        parentColumns = ["sessionId"],
        childColumns = ["sessionId"],
        onDelete = ForeignKey.CASCADE,
    )],
)
data class AlarmSettingsEntity(
    @PrimaryKey val sessionId: String,
    val targetEpochMillis: Long,
    val wakeWindowMinutes: Int,
    val probabilityThreshold: Float,
)

@Entity(
    tableName = "persisted_phone_batches",
    primaryKeys = ["sessionId", "sequence"],
)
data class PersistedPhoneBatchEntity(
    val sessionId: String,
    val sequence: Long,
    val persistedAtEpochMillis: Long,
)

/** Raw samples are stored only while they belong to an unfinished feature window. */
@Entity(tableName = "feature_pipeline_checkpoints")
data class FeaturePipelineCheckpointEntity(
    @PrimaryKey val sessionId: String,
    val currentStartEpochMillis: Long?,
    val lastTimestampEpochMillis: Long?,
    val pendingSamplesPayload: ByteArray,
    val baselinePayload: ByteArray,
)

@Entity(tableName = "pending_watch_acknowledgements", primaryKeys = ["sessionId", "sequence"])
data class PendingWatchAcknowledgementEntity(
    val sessionId: String,
    val sequence: Long,
)

@Entity(
    tableName = "feature_epochs",
    primaryKeys = ["sessionId", "startEpochMillis"],
    indices = [Index(value = ["sessionId", "endEpochMillis"], unique = true)],
)
data class FeatureEpochEntity(
    val sessionId: String,
    val startEpochMillis: Long,
    val endEpochMillis: Long,
    val values: FloatArray,
    val validForInference: Boolean,
)

@Entity(tableName = "prediction_events", indices = [Index(value = ["sessionId", "occurredAtEpochMillis"])])
data class PredictionEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val occurredAtEpochMillis: Long,
    val probability: Float?,
    val validInput: Boolean,
)

@Entity(tableName = "alarm_events", indices = [Index(value = ["sessionId", "occurredAtEpochMillis"])])
data class AlarmEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val occurredAtEpochMillis: Long,
    val reason: String,
)

@Entity(tableName = "refreshed_feeling_feedback", indices = [Index(value = ["sessionId"], unique = true)])
data class RefreshedFeelingFeedbackEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val rating: Int,
    val recordedAtEpochMillis: Long,
)

@Dao
interface SleepSessionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(session: SleepSessionEntity)

    @Query(
        "SELECT s.sessionId, s.startedAtEpochMillis, s.endedAtEpochMillis, s.status, s.researchRetentionEnabled, " +
            "a.targetEpochMillis, a.wakeWindowMinutes, a.probabilityThreshold " +
            "FROM sleep_sessions s INNER JOIN alarm_settings a ON s.sessionId = a.sessionId " +
            "WHERE s.sessionId = :sessionId",
    )
    fun observe(sessionId: String): Flow<SessionWithAlarmSettings?>
}

data class SessionWithAlarmSettings(
    val sessionId: String,
    val startedAtEpochMillis: Long,
    val endedAtEpochMillis: Long?,
    val status: String,
    val researchRetentionEnabled: Boolean,
    val targetEpochMillis: Long,
    val wakeWindowMinutes: Int,
    val probabilityThreshold: Float,
)

@Dao
interface AlarmSettingsDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(settings: AlarmSettingsEntity)
}

@Dao
interface PersistedBatchDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertIgnore(batch: PersistedPhoneBatchEntity): Long

    @Query("SELECT COUNT(*) FROM persisted_phone_batches WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@Dao
interface FeatureEpochDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(epochs: List<FeatureEpochEntity>)

    @Query("SELECT COUNT(*) FROM feature_epochs WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@Dao
interface FeaturePipelineCheckpointDao {
    @Query("SELECT * FROM feature_pipeline_checkpoints WHERE sessionId = :sessionId")
    suspend fun checkpoint(sessionId: String): FeaturePipelineCheckpointEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(checkpoint: FeaturePipelineCheckpointEntity)

    @Query("SELECT COUNT(*) FROM feature_pipeline_checkpoints WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@Dao
interface WatchAcknowledgementDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertIgnore(acknowledgement: PendingWatchAcknowledgementEntity): Long

    @Query("SELECT * FROM pending_watch_acknowledgements ORDER BY sessionId, sequence")
    suspend fun pending(): List<PendingWatchAcknowledgementEntity>

    @Query("DELETE FROM pending_watch_acknowledgements WHERE sessionId = :sessionId AND sequence = :sequence")
    suspend fun delete(sessionId: String, sequence: Long)
}

@Dao
interface PredictionEventDao {
    @Insert
    suspend fun insert(event: PredictionEventEntity)

    @Query("SELECT COUNT(*) FROM prediction_events WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@Dao
interface AlarmEventDao {
    @Insert
    suspend fun insert(event: AlarmEventEntity)

    @Query("SELECT COUNT(*) FROM alarm_events WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@Dao
interface RefreshedFeelingFeedbackDao {
    @Insert
    suspend fun insert(feedback: RefreshedFeelingFeedbackEntity)

    @Query("SELECT COUNT(*) FROM refreshed_feeling_feedback WHERE sessionId = :sessionId")
    fun countForSession(sessionId: String): Int
}

@TypeConverters(FloatArrayConverter::class)
@Database(
    entities = [
        SleepSessionEntity::class,
        AlarmSettingsEntity::class,
        PersistedPhoneBatchEntity::class,
        FeaturePipelineCheckpointEntity::class,
        PendingWatchAcknowledgementEntity::class,
        FeatureEpochEntity::class,
        PredictionEventEntity::class,
        AlarmEventEntity::class,
        RefreshedFeelingFeedbackEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun sleepSessionDao(): SleepSessionDao
    abstract fun alarmSettingsDao(): AlarmSettingsDao
    abstract fun persistedBatchDao(): PersistedBatchDao
    abstract fun featureEpochDao(): FeatureEpochDao
    abstract fun featurePipelineCheckpointDao(): FeaturePipelineCheckpointDao
    abstract fun watchAcknowledgementDao(): WatchAcknowledgementDao
    abstract fun predictionEventDao(): PredictionEventDao
    abstract fun alarmEventDao(): AlarmEventDao
    abstract fun refreshedFeelingFeedbackDao(): RefreshedFeelingFeedbackDao
}

class FloatArrayConverter {
    @TypeConverter
    fun toBytes(values: FloatArray): ByteArray = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
        .apply { values.forEach(::putFloat) }
        .array()

    @TypeConverter
    fun fromBytes(bytes: ByteArray): FloatArray {
        require(bytes.size % Float.SIZE_BYTES == 0) { "Feature values must be a whole number of floats." }
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).let { buffer ->
            FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float }
        }
    }
}
