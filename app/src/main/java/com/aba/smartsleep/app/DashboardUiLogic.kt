package com.aba.smartsleep.app

import java.util.Calendar
import java.util.TimeZone

enum class DashboardTab(val label: String) {
    LIVE("Live"),
    ALARM("Alarm"),
    HISTORY("History"),
}

val SUPPORTED_WAKE_WINDOWS = listOf(15, 30, 45)

fun isSupportedWakeWindow(minutes: Int): Boolean = minutes in SUPPORTED_WAKE_WINDOWS

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
    val latestProbability: Float? = null,
)
