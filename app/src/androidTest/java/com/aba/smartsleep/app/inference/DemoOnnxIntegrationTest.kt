package com.aba.smartsleep.app.inference

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DemoOnnxIntegrationTest {
    @Test
    fun bundledModelScoresDemoWindowAboveAlarmThreshold() {
        val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
        val model = requireNotNull(OnnxProbabilityModel.fromAssetsOrNull(assets))
        try {
            val result = DemoInferenceRunner(LiveInferenceProcessor(model)).run()

            assertTrue("Bundled ONNX model did not return a prediction", result.inferenceAvailable)
            assertTrue("Demo probability ${result.probability} did not cross the alarm threshold", result.aboveAlarmThreshold)
        } finally {
            (model as? AutoCloseable)?.close()
        }
    }
}
