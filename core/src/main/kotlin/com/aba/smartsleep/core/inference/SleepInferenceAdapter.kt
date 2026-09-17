package com.aba.smartsleep.core.inference

import kotlin.math.ln

/** Frozen deployment metadata for the validation-selected MESA-transfer candidate. */
object FrozenSleepModelContract {
    const val sequenceEpochs = 10
    const val featureCount = 8
    const val threshold = 0.37148505f

    val featureNames = listOf(
        "activity_count",
        "activity_availability",
        "heart_rate_mean",
        "heart_rate_standard_deviation",
        "heart_rate_availability",
        "elapsed_hours",
        "clock_sin",
        "clock_cos",
    )

    internal val center = floatArrayOf(0f, 1f, 64.166664f, 0.8164966f, 1f, 3.1666667f, 0.77920127f, 0.6118147f)
    internal val scale = floatArrayOf(1f, 1f, 11.309521f, 0.9243156f, 1f, 3.6249998f, 0.5359808f, 0.7256329f)
}

/** Raw common-schema values. This deliberately cannot accept FeatureEpoch.modelInput(). */
data class SharedFeatureEpoch(
    val activityCount: Float,
    val activityAvailability: Float,
    val heartRateMean: Float,
    val heartRateStandardDeviation: Float,
    val heartRateAvailability: Float,
    val elapsedHours: Float,
    val clockSin: Float,
    val clockCos: Float,
    val validForInference: Boolean,
)

sealed interface InferenceResult {
    data class Prediction(val probability: Float, val aboveThreshold: Boolean) : InferenceResult
    data object Unavailable : InferenceResult
}

fun interface ProbabilityModel {
    fun predict(shape: LongArray, values: FloatArray): Float
}

/** Prepares the exact frozen tensor contract; any invalid input/runtime result fails closed. */
class SleepInferenceAdapter(private val model: ProbabilityModel?) {
    fun predict(epochs: List<SharedFeatureEpoch>): InferenceResult {
        if (model == null || epochs.size != FrozenSleepModelContract.sequenceEpochs ||
            epochs.any { !it.validForInference }
        ) return InferenceResult.Unavailable

        val tensor = epochs.flatMap { epoch ->
            listOf(
                ln(epoch.activityCount.coerceAtLeast(0f).toDouble() + 1.0).toFloat(),
                epoch.activityAvailability,
                epoch.heartRateMean,
                epoch.heartRateStandardDeviation,
                epoch.heartRateAvailability,
                epoch.elapsedHours,
                epoch.clockSin,
                epoch.clockCos,
            )
        }.mapIndexed { index, value ->
            (value - FrozenSleepModelContract.center[index % FrozenSleepModelContract.featureCount]) /
                FrozenSleepModelContract.scale[index % FrozenSleepModelContract.featureCount]
        }.toFloatArray()
        if (tensor.any { !it.isFinite() }) return InferenceResult.Unavailable

        val probability = try {
            model.predict(longArrayOf(1, FrozenSleepModelContract.sequenceEpochs.toLong(), FrozenSleepModelContract.featureCount.toLong()), tensor)
        } catch (_: Throwable) {
            return InferenceResult.Unavailable
        }
        if (!probability.isFinite() || probability !in 0f..1f) return InferenceResult.Unavailable
        return InferenceResult.Prediction(probability, probability >= FrozenSleepModelContract.threshold)
    }
}
