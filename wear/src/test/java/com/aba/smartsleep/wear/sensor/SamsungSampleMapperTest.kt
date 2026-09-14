package com.aba.smartsleep.wear.sensor

import com.aba.smartsleep.core.model.SensorQuality
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SamsungSampleMapperTest {
    @Test
    fun `maps off body heart rate status to off body quality`() {
        val sample = SamsungSampleMapper.heartRate(
            timestamp = 1_000,
            bpm = 0f,
            ibi = null,
            status = -3,
        )

        assertEquals(1_000, sample.timestampEpochMillis)
        assertEquals(0f, sample.heartRateBpm)
        assertNull(sample.ibiMillis)
        assertEquals(SensorQuality.OFF_BODY, sample.quality)
    }

    @Test
    fun `retains a valid heart rate IBI`() {
        val sample = SamsungSampleMapper.heartRate(
            timestamp = 2_000,
            bpm = 62f,
            ibi = 967,
            status = 0,
        )

        assertEquals(62f, sample.heartRateBpm)
        assertEquals(967, sample.ibiMillis)
        assertEquals(SensorQuality.VALID, sample.quality)
    }

    @Test
    fun `preserves raw accelerometer axes`() {
        val sample = SamsungSampleMapper.accelerometer(
            timestamp = 3_000,
            x = 11,
            y = -12,
            z = 13,
        )

        assertEquals(11f, sample.accelX)
        assertEquals(-12f, sample.accelY)
        assertEquals(13f, sample.accelZ)
        assertEquals(SensorQuality.VALID, sample.quality)
    }
}
