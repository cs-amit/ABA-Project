package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import com.aba.smartsleep.core.inference.FrozenSleepModelContract

const val DEMO_BUTTON_LABEL = "Run 10-epoch demo"

data class DemoInferenceResult(
    val epochsProcessed: Int,
    val probability: Float,
    val aboveAlarmThreshold: Boolean,
    val latestEpoch: FeatureEpoch,
    val inferenceAvailable: Boolean,
) {
    val isDemo: Boolean = true
    val label: String = DemoInferenceScenario.LABEL

    /** The demo is explanatory only and cannot schedule or sound an alarm. */
    val shouldTriggerRealAlarm: Boolean = false
}

/** A deterministic, visibly synthetic sleep-like window for classroom demonstrations. */
object DemoInferenceScenario {
    const val SESSION_ID = "demo-synthetic-sleep"
    const val LABEL = "Demo: synthetic sleep-like window"
    private const val BASE_EPOCH_MILLIS = 1_758_219_300_000L // 2025-09-18 23:45 Asia/Kolkata
    private const val EPOCH_MILLIS = 30_000L

    fun epochs(): List<FeatureEpoch> = List(FrozenSleepModelContract.sequenceEpochs) { index ->
        val raw = FloatArray(FeatureValue.entries.size).apply {
            this[FeatureValue.ACTIVITY_COUNT.index] = 0f
            this[FeatureValue.HEART_RATE_MEAN.index] = 60f
            this[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] = 1f
            this[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] = 1f
            this[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] = 1f
        }
        val start = BASE_EPOCH_MILLIS + index * EPOCH_MILLIS
        FeatureEpoch(SESSION_ID, start, start + EPOCH_MILLIS, raw, true, raw.copyOf())
    }
}

class DemoInferenceRunner(private val processor: LiveInferenceProcessor) {
    fun run(): DemoInferenceResult {
        val epochs = DemoInferenceScenario.epochs()
        var probability = 0f
        var aboveThreshold = false
        var inferenceAvailable = false
        epochs.forEach { epoch ->
            processor.accept(epoch)?.let { prediction ->
                probability = prediction.probability
                aboveThreshold = prediction.aboveThreshold
                inferenceAvailable = true
            }
        }
        return DemoInferenceResult(
            epochsProcessed = epochs.size,
            probability = probability,
            aboveAlarmThreshold = aboveThreshold,
            latestEpoch = epochs.last(),
            inferenceAvailable = inferenceAvailable,
        )
    }
}
