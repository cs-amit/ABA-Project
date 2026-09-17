package com.aba.smartsleep.core.inference

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class SleepInferenceAdapterTest {
    @Test
    fun `uses frozen eight feature order and robust scaling`() {
        var observedShape: LongArray? = null
        var observedValues: FloatArray? = null
        val adapter = SleepInferenceAdapter { shape, values ->
            observedShape = shape
            observedValues = values
            0.5f
        }

        val result = adapter.predict(List(10) { validEpoch() })

        assertEquals(InferenceResult.Prediction(0.5f, true), result)
        assertContentEquals(longArrayOf(1, 10, 8), observedShape)
        floatArrayOf(kotlin.math.ln(10.0).toFloat(), 0f, 0f, 0f, 0f, 0f, 0f, 0f)
            .forEachIndexed { index, expected ->
                assertEquals(expected, observedValues!![index], 0.0001f)
            }
    }

    @Test
    fun `refuses incomplete or low quality sequences without invoking model`() {
        var calls = 0
        val adapter = SleepInferenceAdapter { _, _ -> calls++; 0.9f }

        assertEquals(InferenceResult.Unavailable, adapter.predict(List(9) { validEpoch() }))
        assertEquals(
            InferenceResult.Unavailable,
            adapter.predict(List(10) { validEpoch().copy(validForInference = false) }),
        )
        assertEquals(0, calls)
    }

    @Test
    fun `fails closed for invalid values model failures and invalid probabilities`() {
        val invalid = List(10) { validEpoch().copy(heartRateMean = Float.NaN) }
        assertEquals(InferenceResult.Unavailable, SleepInferenceAdapter { _, _ -> 0.9f }.predict(invalid))
        assertEquals(
            InferenceResult.Unavailable,
            SleepInferenceAdapter { _, _ -> error("runtime failure") }.predict(List(10) { validEpoch() }),
        )
        assertEquals(
            InferenceResult.Unavailable,
            SleepInferenceAdapter { _, _ -> 2f }.predict(List(10) { validEpoch() }),
        )
    }

    @Test
    fun `threshold is frozen validation threshold`() {
        assertEquals(0.37148505f, FrozenSleepModelContract.threshold)
        assertEquals(
            listOf(
                "activity_count", "activity_availability", "heart_rate_mean",
                "heart_rate_standard_deviation", "heart_rate_availability", "elapsed_hours",
                "clock_sin", "clock_cos",
            ),
            FrozenSleepModelContract.featureNames,
        )
    }

    private fun validEpoch() = SharedFeatureEpoch(
        activityCount = 9f,
        activityAvailability = 1f,
        heartRateMean = 64.166664f,
        heartRateStandardDeviation = 0.8164966f,
        heartRateAvailability = 1f,
        elapsedHours = 3.1666667f,
        clockSin = 0.77920127f,
        clockCos = 0.6118147f,
        validForInference = true,
    )
}
