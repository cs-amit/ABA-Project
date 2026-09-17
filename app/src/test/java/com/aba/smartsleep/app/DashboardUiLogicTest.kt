package com.aba.smartsleep.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

class DashboardUiLogicTest {
    @Test
    fun navigationTabsHaveStableLabels() {
        assertEquals(listOf("Live", "Alarm", "History"), DashboardTab.entries.map { it.label })
    }

    @Test
    fun onlyClassroomWakeWindowsAreAccepted() {
        listOf(15, 30, 45).forEach { assertTrue(isSupportedWakeWindow(it)) }
        listOf(0, 1, 5, 10, 20, 60).forEach { assertFalse(isSupportedWakeWindow(it)) }
    }

    @Test
    fun selectedTimeTodayRollsToTomorrowWhenAlreadyPassed() {
        val zone = TimeZone.getTimeZone("UTC")
        val now = Calendar.getInstance(zone).apply {
            set(2026, Calendar.SEPTEMBER, 18, 22, 0, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

        val trigger = nextAlarmEpochMillis(now, 21, 30, zone)
        val expected = Calendar.getInstance(zone).apply {
            set(2026, Calendar.SEPTEMBER, 19, 21, 30, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

        assertEquals(expected, trigger)
    }

    @Test
    fun selectedFutureTimeStaysToday() {
        val zone = TimeZone.getTimeZone("UTC")
        val now = Calendar.getInstance(zone).apply {
            set(2026, Calendar.SEPTEMBER, 18, 20, 0, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

        val trigger = nextAlarmEpochMillis(now, 21, 30, zone)

        assertEquals(21, Calendar.getInstance(zone).apply { timeInMillis = trigger }.get(Calendar.HOUR_OF_DAY))
        assertEquals(Calendar.SEPTEMBER, Calendar.getInstance(zone).apply { timeInMillis = trigger }.get(Calendar.MONTH))
        assertEquals(18, Calendar.getInstance(zone).apply { timeInMillis = trigger }.get(Calendar.DAY_OF_MONTH))
    }
}
