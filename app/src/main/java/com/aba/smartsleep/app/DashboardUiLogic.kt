package com.aba.smartsleep.app

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import java.util.Calendar
import java.util.TimeZone
import kotlin.math.roundToInt

enum class DashboardTab(val label: String) {
    LIVE("Live"),
    ALARM("Alarm"),
    HISTORY("History"),
}

val SUPPORTED_WAKE_WINDOWS = listOf(15, 30, 45)

fun isSupportedWakeWindow(minutes: Int): Boolean = minutes in SUPPORTED_WAKE_WINDOWS

fun notificationPermissionNeeded(sdkInt: Int, permissionGranted: Boolean): Boolean =
    sdkInt >= 33 && !permissionGranted

data class EpochSummary(
    val startEpochMillis: Long,
    val endEpochMillis: Long,
    val activityCount: Float,
    val heartRateMean: Float,
    val heartRateVariability: Float,
    val motionCoveragePercent: Int,
    val heartRateCoveragePercent: Int,
    val validForInference: Boolean,
)

fun FeatureEpoch.toEpochSummary(): EpochSummary = EpochSummary(
    startEpochMillis = startEpochMillis,
    endEpochMillis = endEpochMillis,
    activityCount = rawValues[FeatureValue.ACTIVITY_COUNT.index],
    heartRateMean = rawValues[FeatureValue.HEART_RATE_MEAN.index],
    heartRateVariability = rawValues[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index],
    motionCoveragePercent = (rawValues[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] * 100f).roundToInt().coerceIn(0, 100),
    heartRateCoveragePercent = (rawValues[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] * 100f).roundToInt().coerceIn(0, 100),
    validForInference = validForInference,
)

fun nextAlarmEpochMillis(
    nowEpochMillis: Long,
    hourOfDay: Int,
    minute: Int,
    timeZone: TimeZone = TimeZone.getDefault(),
): Long {
    require(hourOfDay in 0..23)
    require(minute in 0..59)
    val selected = Calendar.getInstance(timeZone).apply {
        timeInMillis = nowEpochMillis
        set(Calendar.HOUR_OF_DAY, hourOfDay)
        set(Calendar.MINUTE, minute)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
        if (timeInMillis <= nowEpochMillis) add(Calendar.DAY_OF_YEAR, 1)
    }
    return selected.timeInMillis
}

data class DashboardState(
    val watchLinked: Boolean = false,
    val sessionId: String? = null,
    val batchesReceived: Int = 0,
    val samplesReceived: Int = 0,
    val epochsGenerated: Int = 0,
    val epochsInCurrentWindow: Int = 0,
    val latestEpoch: EpochSummary? = null,
    val latestProbability: Float? = null,
)
