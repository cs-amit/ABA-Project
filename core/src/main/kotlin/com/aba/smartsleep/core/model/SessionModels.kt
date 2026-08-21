package com.aba.smartsleep.core.model

data class AlarmSettings(
    val targetEpochMillis: Long,
    val wakeWindowMinutes: Int,
    val probabilityThreshold: Float = 0.70f,
) {
    init {
        require(wakeWindowMinutes in setOf(15, 30, 45))
        require(probabilityThreshold in 0f..1f)
    }

    val wakeWindowStartEpochMillis: Long
        get() = targetEpochMillis - wakeWindowMinutes * 60_000L
}

data class SleepSession(
    val sessionId: String,
    val settings: AlarmSettings,
    val startedAtEpochMillis: Long,
    val status: SessionStatus = SessionStatus.ACTIVE,
    val endedAtEpochMillis: Long? = null,
)

enum class SessionStatus {
    SCHEDULED,
    ACTIVE,
    COMPLETED,
    CANCELLED,
}

data class SensorSample(
    val timestampEpochMillis: Long,
    val accelX: Float?,
    val accelY: Float?,
    val accelZ: Float?,
    val heartRateBpm: Float?,
    val ibiMillis: Int?,
    val quality: SensorQuality,
)

class SensorBatch(
    val sessionId: String,
    samples: List<SensorSample>,
) {
    val samples: List<SensorSample> = samples.toList()

    init {
        require(this.samples.zipWithNext().all { (first, second) ->
            first.timestampEpochMillis <= second.timestampEpochMillis
        })
    }

    override fun equals(other: Any?): Boolean =
        other is SensorBatch && sessionId == other.sessionId && samples == other.samples

    override fun hashCode(): Int = 31 * sessionId.hashCode() + samples.hashCode()
}

enum class SensorQuality {
    VALID,
    DEGRADED,
    OFF_BODY,
    UNAVAILABLE,
}

enum class WearableCapability {
    ACCELEROMETER,
    HEART_RATE_WITH_IBI,
    HAPTICS,
}
