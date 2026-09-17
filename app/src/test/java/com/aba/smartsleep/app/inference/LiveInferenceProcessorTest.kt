package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import com.aba.smartsleep.core.inference.ProbabilityModel
import org.junit.Assert.assertEquals
import org.junit.Test

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

    private fun epoch(index: Int, valid: Boolean = true): FeatureEpoch {
        val raw = FloatArray(FeatureValue.entries.size)
        raw[FeatureValue.ACTIVITY_COUNT.index] = 4f
        raw[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] = 1f
        raw[FeatureValue.HEART_RATE_MEAN.index] = 64f
        raw[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] = 1f
        raw[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] = 1f
        val start = index * 30_000L
        return FeatureEpoch("s1", start, start + 30_000L, raw, valid, raw)
    }
}
