package com.aba.smartsleep.app

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

class DashboardUiLogicTest {
    @Test
    fun `small nonzero predictions retain percentage precision`() {
        assertEquals("0.63%", formatSleepProbability(0.00626710057f))
        assertEquals("0.42%", formatSleepProbability(0.00420469046f))
        assertEquals("0.81%", formatSleepProbability(0.00814169645f))
    }

    @Test
    fun `tiny positive predictions remain distinct from an exact zero`() {
        assertEquals("<0.01%", formatSleepProbability(0.00001f))
        assertEquals("0.00%", formatSleepProbability(0f))
        assertEquals("100.00%", formatSleepProbability(1f))
    }

    @Test
    fun `feature epoch is converted to a readable dashboard summary`() {
        val raw = FloatArray(FeatureValue.entries.size).apply {
            this[FeatureValue.ACTIVITY_COUNT.index] = 7f
            this[FeatureValue.HEART_RATE_MEAN.index] = 63.5f
            this[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] = 2.25f
            this[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] = .9f
            this[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] = .8f
        }

        val summary = FeatureEpoch("session", 1_000L, 31_000L, raw, true, raw).toEpochSummary()

        assertEquals(7f, summary.activityCount)
        assertEquals(63.5f, summary.heartRateMean)
        assertEquals(2.25f, summary.heartRateVariability)
        assertEquals(90, summary.motionCoveragePercent)
        assertEquals(80, summary.heartRateCoveragePercent)
        assertTrue(summary.validForInference)
    }

    @Test
    fun `notification action is needed only for an ungranted runtime permission`() {
        assertTrue(notificationPermissionNeeded(35, permissionGranted = false))
        assertFalse(notificationPermissionNeeded(35, permissionGranted = true))
        assertFalse(notificationPermissionNeeded(32, permissionGranted = false))
    }

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
