package com.aba.smartsleep.wear.sensor

import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.model.WearableCapability
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class SensorCaptureStateTest {
    @Test
    fun `RetainedSampleHandoff preserves captured sample emitted before transport collection begins`() = runBlocking {
        val handoff = RetainedSampleHandoff(capacity = 1)
        val capturedSample = SensorSample(
            timestampEpochMillis = 1_000,
            accelX = 1f,
            accelY = 2f,
            accelZ = 3f,
            heartRateBpm = null,
            ibiMillis = null,
            quality = SensorQuality.VALID,
        )

        handoff.emit(capturedSample)

        assertEquals(
            capturedSample,
            withTimeout(1_000) { handoff.samples().first() },
        )
    }

    @Test
    fun `CaptureCapabilityState clears sensor capabilities and tracker support after capture failure`() {
        val state = CaptureCapabilityState(setOf(WearableCapability.HAPTICS))
        state.onServiceConnected(
            setOf(
                HealthTrackerType.ACCELEROMETER_CONTINUOUS,
                HealthTrackerType.HEART_RATE_CONTINUOUS,
            ),
        )
        assertEquals(
            setOf(
                WearableCapability.HAPTICS,
                WearableCapability.ACCELEROMETER,
                WearableCapability.HEART_RATE_WITH_IBI,
            ),
            state.capabilities(),
        )

        state.onCaptureFailure()

        assertEquals(setOf(WearableCapability.HAPTICS), state.capabilities())
        assertFalse(state.supports(HealthTrackerType.ACCELEROMETER_CONTINUOUS))
        assertFalse(state.supports(HealthTrackerType.HEART_RATE_CONTINUOUS))
    }
}
