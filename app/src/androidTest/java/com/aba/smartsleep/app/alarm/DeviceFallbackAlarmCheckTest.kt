package com.aba.smartsleep.app.alarm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DeviceFallbackAlarmCheckTest {
    @Test
    fun schedulesFutureFallbackAlarmFromMandatoryTargetArgument() {
        val targetArgument = InstrumentationRegistry.getArguments().getString(TARGET_EPOCH_MILLIS_ARGUMENT)
            ?: run {
                assumeTrue("targetEpochMillis instrumentation argument is required", false)
                return
            }

        val targetEpochMillis = targetArgument.toLongOrNull()
            ?: throw AssertionError("targetEpochMillis must be an epoch-millis value")
        assertTrue(
            "targetEpochMillis must be in the future",
            targetEpochMillis > System.currentTimeMillis(),
        )

        AndroidAlarmGateway(InstrumentationRegistry.getInstrumentation().targetContext).schedule(
            AlarmRequest(SESSION_ID, targetEpochMillis),
        )
    }

    private companion object {
        const val SESSION_ID = "device-alarm-check"
        const val TARGET_EPOCH_MILLIS_ARGUMENT = "targetEpochMillis"
    }
}
