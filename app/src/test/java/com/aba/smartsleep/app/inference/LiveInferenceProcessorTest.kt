package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import com.aba.smartsleep.core.inference.ProbabilityModel
import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.Instant

class LiveInferenceProcessorTest {
    @Test
    fun `invokes model once after ten contiguous valid epochs`() {
        var calls = 0
        val processor = LiveInferenceProcessor(ProbabilityModel { _, _ -> calls++; 0.6f })

        repeat(10) { index -> processor.accept(epoch(index)) }

        assertEquals(1, calls)
        assertEquals(0.6f, processor.latestPrediction?.probability)
    }

    @Test
    fun `invalid epoch clears live inference window`() {
        var calls = 0
        val processor = LiveInferenceProcessor(ProbabilityModel { _, _ -> calls++; 0.6f })
        repeat(9) { processor.accept(epoch(it)) }
        processor.accept(epoch(9, valid = false))
        processor.accept(epoch(10))

        assertEquals(0, calls)
    }

    @Test
    fun `completed prediction window resets before collecting the next ten epochs`() {
        var calls = 0
        val processor = LiveInferenceProcessor(ProbabilityModel { _, _ -> calls++; 0.6f })
        repeat(10) { processor.accept(epoch(it)) }

        repeat(9) { processor.accept(epoch(it + 10)) }
        assertEquals(1, calls)
        assertEquals(9, processor.currentWindowEpochCount("s1"))

        processor.accept(epoch(19))
        assertEquals(2, calls)
        assertEquals(0, processor.currentWindowEpochCount("s1"))
    }

    @Test
    fun `encodes model clock features in India time`() {
        var modelValues = FloatArray(0)
        val processor = LiveInferenceProcessor(ProbabilityModel { _, values ->
            modelValues = values
            0.6f
        })
        val midnightUtc = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli()

        repeat(10) { index -> processor.accept(epoch(index, baseEpochMillis = midnightUtc)) }

        assertEquals(0.39599103f, modelValues[6], 0.000001f)
        assertEquals(-0.6632672f, modelValues[7], 0.000001f)
    }

    private fun epoch(index: Int, valid: Boolean = true, baseEpochMillis: Long = 0L): FeatureEpoch {
        val raw = FloatArray(FeatureValue.entries.size)
        raw[FeatureValue.ACTIVITY_COUNT.index] = 4f
        raw[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] = 1f
        raw[FeatureValue.HEART_RATE_MEAN.index] = 64f
        raw[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] = 1f
        raw[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] = 1f
        val start = baseEpochMillis + index * 30_000L
        return FeatureEpoch("s1", start, start + 30_000L, raw, valid, raw)
    }
}
