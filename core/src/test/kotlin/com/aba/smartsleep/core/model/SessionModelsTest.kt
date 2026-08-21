package com.aba.smartsleep.core.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SessionModelsTest {
    @Test
    fun `wake window must end at target alarm`() {
        val settings = AlarmSettings(targetEpochMillis = 1_000_000, wakeWindowMinutes = 30)

        assertEquals(-800_000, settings.wakeWindowStartEpochMillis)
    }

    @Test
    fun `sensor batch rejects unordered samples`() {
        assertFailsWith<IllegalArgumentException> {
            SensorBatch("session", listOf(sampleAt(20), sampleAt(10)))
        }
    }

    @Test
    fun `sensor batch keeps ordered samples when original list is mutated`() {
        val originalSamples = mutableListOf(sampleAt(10), sampleAt(20))
        val batch = SensorBatch("session", originalSamples)

        originalSamples[0] = sampleAt(30)

        assertEquals(listOf(10L, 20L), batch.samples.map { it.timestampEpochMillis })
    }

    @Test
    fun `alarm settings reject unsupported wake windows`() {
        assertFailsWith<IllegalArgumentException> {
            AlarmSettings(targetEpochMillis = 1_000_000, wakeWindowMinutes = 20)
        }
    }

    @Test
    fun `alarm settings reject probabilities outside unit interval`() {
        assertFailsWith<IllegalArgumentException> {
            AlarmSettings(targetEpochMillis = 1_000_000, wakeWindowMinutes = 30, probabilityThreshold = 1.1f)
        }
    }

    private fun sampleAt(timestampEpochMillis: Long) = SensorSample(
        timestampEpochMillis = timestampEpochMillis,
        accelX = null,
        accelY = null,
        accelZ = null,
        heartRateBpm = null,
        ibiMillis = null,
        quality = SensorQuality.UNAVAILABLE,
    )
}
