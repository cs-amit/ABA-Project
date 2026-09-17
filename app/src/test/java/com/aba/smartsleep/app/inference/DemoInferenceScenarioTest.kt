package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.features.FeatureValue
import com.aba.smartsleep.core.inference.ProbabilityModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoInferenceScenarioTest {
    @Test
    fun `demo creates ten contiguous valid sleep-like epochs`() {
        val epochs = DemoInferenceScenario.epochs()

        assertEquals(10, epochs.size)
        assertTrue(epochs.all { it.sessionId == DemoInferenceScenario.SESSION_ID })
        assertTrue(epochs.all { it.validForInference })
        assertTrue(epochs.zipWithNext().all { (first, second) -> first.endEpochMillis == second.startEpochMillis })
        assertTrue(epochs.all { it.rawValues[FeatureValue.ACTIVITY_COUNT.index] == 0f })
        assertTrue(epochs.all { it.rawValues[FeatureValue.HEART_RATE_MEAN.index] == 60f })
        assertTrue(epochs.all { it.rawValues[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] == 1f })
        assertTrue(epochs.all { it.rawValues[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] == 1f })
        assertTrue(epochs.all { it.rawValues[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] == 1f })
    }

    @Test
    fun `demo runner sends all ten epochs through live processor and reports threshold result`() {
        var modelCalls = 0
        val processor = LiveInferenceProcessor(ProbabilityModel { _, _ ->
            modelCalls += 1
            0.82f
        })

        val result = DemoInferenceRunner(processor).run()

        assertEquals(1, modelCalls)
        assertEquals(10, result.epochsProcessed)
        assertEquals(0.82f, result.probability, 0f)
        assertTrue(result.aboveAlarmThreshold)
        assertTrue(result.isDemo)
        assertEquals("Demo: synthetic sleep-like window", result.label)
    }

    @Test
    fun `demo threshold result remains informational and never requests an alarm`() {
        val processor = LiveInferenceProcessor(ProbabilityModel { _, _ -> 0.82f })

        val result = DemoInferenceRunner(processor).run()

        assertFalse(result.shouldTriggerRealAlarm)
        assertEquals("Run 10-epoch demo", DEMO_BUTTON_LABEL)
    }
}
