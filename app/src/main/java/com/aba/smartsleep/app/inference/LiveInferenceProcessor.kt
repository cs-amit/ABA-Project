package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.features.FeatureEpoch
import com.aba.smartsleep.core.features.FeatureValue
import com.aba.smartsleep.core.inference.InferenceResult
import com.aba.smartsleep.core.inference.ProbabilityModel
import com.aba.smartsleep.core.inference.SharedFeatureEpoch
import com.aba.smartsleep.core.inference.SleepInferenceAdapter
import java.time.Instant
import java.time.ZoneId
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Passive classroom-MVP bridge from completed phone feature epochs to the frozen model. */
class LiveInferenceProcessor(model: ProbabilityModel?) {
    private val adapter = SleepInferenceAdapter(model)
    private val windows = mutableMapOf<String, ArrayDeque<SharedFeatureEpoch>>()
    private val sessionStarts = mutableMapOf<String, Long>()
    private val lastEnds = mutableMapOf<String, Long>()
    var latestPrediction: InferenceResult.Prediction? = null
        private set

    fun accept(epoch: FeatureEpoch): InferenceResult.Prediction? {
        val window = windows.getOrPut(epoch.sessionId) { ArrayDeque() }
        val discontinuous = lastEnds[epoch.sessionId]?.let { it != epoch.startEpochMillis } == true
        if (!epoch.validForInference || discontinuous) {
            window.clear()
        }
        if (discontinuous) sessionStarts[epoch.sessionId] = epoch.startEpochMillis
        lastEnds[epoch.sessionId] = epoch.endEpochMillis
        if (!epoch.validForInference) return null
        val start = sessionStarts.getOrPut(epoch.sessionId) { epoch.startEpochMillis }
        window += epoch.toShared(start)
        if (window.size < com.aba.smartsleep.core.inference.FrozenSleepModelContract.sequenceEpochs) return null
        val result = adapter.predict(window.toList()) as? InferenceResult.Prediction
        window.clear()
        if (result != null) latestPrediction = result
        return result
    }

    fun currentWindowEpochCount(sessionId: String): Int = windows[sessionId]?.size ?: 0

    private fun FeatureEpoch.toShared(sessionStart: Long): SharedFeatureEpoch {
        val localTime = Instant.ofEpochMilli(startEpochMillis).atZone(MODEL_ZONE).toLocalTime()
        val angle = 2.0 * PI * localTime.toSecondOfDay() / 86_400.0
        return SharedFeatureEpoch(
            activityCount = rawValues[FeatureValue.ACTIVITY_COUNT.index],
            activityAvailability = rawValues[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index],
            heartRateMean = rawValues[FeatureValue.HEART_RATE_MEAN.index],
            heartRateStandardDeviation = rawValues[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index],
            heartRateAvailability = rawValues[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index],
            elapsedHours = (startEpochMillis - sessionStart) / 3_600_000f,
            clockSin = sin(angle).toFloat(),
            clockCos = cos(angle).toFloat(),
            validForInference = true,
        )
    }

    private companion object {
        val MODEL_ZONE: ZoneId = ZoneId.of("Asia/Kolkata")
    }
}
