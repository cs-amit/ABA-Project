package com.aba.smartsleep.core.features

import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

class FeatureEpoch(
    val sessionId: String,
    val startEpochMillis: Long,
    val endEpochMillis: Long,
    val values: FloatArray,
    val validForInference: Boolean,
    /** Unscaled extraction values retained for the separately frozen deployment scaler. */
    val rawValues: FloatArray = values,
) {
    override fun equals(other: Any?): Boolean = other is FeatureEpoch && sessionId == other.sessionId &&
        startEpochMillis == other.startEpochMillis && endEpochMillis == other.endEpochMillis &&
        values.contentEquals(other.values) && validForInference == other.validForInference && rawValues.contentEquals(other.rawValues)

    override fun hashCode(): Int = arrayOf(sessionId, startEpochMillis, endEpochMillis, values.contentHashCode(), validForInference, rawValues.contentHashCode()).contentHashCode()

    fun modelInput(): FloatArray {
        require(values.size == FeatureValue.entries.size) { "Feature epoch must use the complete extraction contract." }
        return ModelInputFeature.entries.map { values[it.source.index] }.toFloatArray()
    }
}

enum class FeatureValue(val index: Int) {
    ACCEL_MAGNITUDE_MEAN(0), ACCEL_MAGNITUDE_STANDARD_DEVIATION(1), ACCEL_MAGNITUDE_MEDIAN_ABSOLUTE_DEVIATION(2),
    ACTIVITY_COUNT(3), ACCEL_ZERO_CROSSING_RATE(4), HEART_RATE_MEAN(5), HEART_RATE_STANDARD_DEVIATION(6),
    IBI_MEAN(7), IBI_RMSSD(8), ACCEL_VALID_SAMPLE_RATIO(9), HEART_RATE_VALID_SAMPLE_RATIO(10), OFF_BODY_FLAG(11),
}

/** The trainable model tensor contains only features represented by the BIDSleep source data. */
enum class ModelInputFeature(val source: FeatureValue) {
    ACCEL_MAGNITUDE_MEAN(FeatureValue.ACCEL_MAGNITUDE_MEAN),
    ACCEL_MAGNITUDE_STANDARD_DEVIATION(FeatureValue.ACCEL_MAGNITUDE_STANDARD_DEVIATION),
    ACCEL_MAGNITUDE_MEDIAN_ABSOLUTE_DEVIATION(FeatureValue.ACCEL_MAGNITUDE_MEDIAN_ABSOLUTE_DEVIATION),
    ACTIVITY_COUNT(FeatureValue.ACTIVITY_COUNT),
    ACCEL_ZERO_CROSSING_RATE(FeatureValue.ACCEL_ZERO_CROSSING_RATE),
    HEART_RATE_MEAN(FeatureValue.HEART_RATE_MEAN),
    HEART_RATE_STANDARD_DEVIATION(FeatureValue.HEART_RATE_STANDARD_DEVIATION),
    ACCEL_VALID_SAMPLE_RATIO(FeatureValue.ACCEL_VALID_SAMPLE_RATIO),
    HEART_RATE_VALID_SAMPLE_RATIO(FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO),
}

/** Durable state for the current, untransformed window and its causal baseline. */
data class FeaturePipelineState(
    val sessionId: String,
    val currentStartEpochMillis: Long?,
    val lastTimestampEpochMillis: Long?,
    val pendingSamples: List<SensorSample>,
    val priorValidRawValues: List<FloatArray>,
)

class FeaturePipeline(
    private val sessionId: String = "",
    private val epochMillis: Long = DEFAULT_EPOCH_MILLIS,
    private val minimumValidSampleRatio: Float = MINIMUM_VALID_SAMPLE_RATIO,
) {
    init {
        require(epochMillis > 0)
        require(minimumValidSampleRatio in 0f..1f)
    }

    val completedEpochs = mutableListOf<FeatureEpoch>()
    var droppedFinalizedSampleCount: Int = 0
        private set
    private val samples = mutableListOf<SensorSample>()
    private val priorValidRawValues = mutableListOf<FloatArray>()
    private var currentStartEpochMillis: Long? = null
    private var lastTimestampEpochMillis: Long? = null

    fun append(sample: SensorSample): List<FeatureEpoch> {
        lastTimestampEpochMillis?.let { lastTimestamp ->
            if (sample.timestampEpochMillis < lastTimestamp) {
                if (sample.timestampEpochMillis < requireNotNull(currentStartEpochMillis)) {
                    droppedFinalizedSampleCount += 1
                    return emptyList()
                }
                samples += sample
                samples.sortBy(SensorSample::timestampEpochMillis)
                return emptyList()
            }
        }
        lastTimestampEpochMillis = sample.timestampEpochMillis
        var openStart = currentStartEpochMillis ?: (Math.floorDiv(sample.timestampEpochMillis, epochMillis) * epochMillis).also {
            currentStartEpochMillis = it
        }
        val emitted = mutableListOf<FeatureEpoch>()
        while (sample.timestampEpochMillis >= openStart + epochMillis) {
            emitted += completeEpoch(openStart, openStart + epochMillis)
            openStart += epochMillis
            currentStartEpochMillis = openStart
        }
        samples += sample
        return emitted
    }

    fun consume(newSamples: Iterable<SensorSample>): List<FeatureEpoch> = newSamples.flatMap(::append)

    fun snapshot(): FeaturePipelineState = FeaturePipelineState(
        sessionId, currentStartEpochMillis, lastTimestampEpochMillis, samples.toList(), priorValidRawValues.map(FloatArray::copyOf),
    )

    private fun completeEpoch(start: Long, end: Long): FeatureEpoch {
        val inEpoch = samples.filter { it.timestampEpochMillis in start until end }
        samples.removeAll(inEpoch.toSet())
        val extraction = extract(inEpoch)
        val qualitySufficient = extraction.isSufficient()
        val inferenceReady = qualitySufficient && priorValidRawValues.isNotEmpty()
        val values = normalize(extraction.values)
        if (qualitySufficient) priorValidRawValues += extraction.values.copyOf()
        return FeatureEpoch(sessionId, start, end, values, inferenceReady, extraction.values.copyOf()).also(completedEpochs::add)
    }

    private fun extract(epoch: List<SensorSample>): Extraction {
        val accelObserved = epoch.filter { it.accelX != null || it.accelY != null || it.accelZ != null }
        val accelValid = accelObserved.filter { it.quality == SensorQuality.VALID && it.accelX != null && it.accelY != null && it.accelZ != null }
        val hrObserved = epoch.filter { it.heartRateBpm != null }
        val hrValid = hrObserved.filter { it.quality == SensorQuality.VALID }
        val ibiValid = epoch.filter { it.quality == SensorQuality.VALID && it.ibiMillis != null }
        val magnitudes = accelValid.map { sqrt(it.accelX!!.toDouble() * it.accelX + it.accelY!!.toDouble() * it.accelY + it.accelZ!!.toDouble() * it.accelZ).toFloat() }
        val heartRates = hrValid.map { requireNotNull(it.heartRateBpm) }
        val ibis = ibiValid.map { requireNotNull(it.ibiMillis).toFloat() }
        val accelMean = magnitudes.meanOrZero()
        return Extraction(
            FloatArray(FeatureValue.entries.size).also { values ->
                values[FeatureValue.ACCEL_MAGNITUDE_MEAN.index] = accelMean
                values[FeatureValue.ACCEL_MAGNITUDE_STANDARD_DEVIATION.index] = magnitudes.standardDeviation()
                values[FeatureValue.ACCEL_MAGNITUDE_MEDIAN_ABSOLUTE_DEVIATION.index] = magnitudes.medianAbsoluteDeviation()
                values[FeatureValue.ACTIVITY_COUNT.index] = magnitudes.zipWithNext().count { (a, b) -> abs(b - a) >= ACTIVITY_DELTA_THRESHOLD }.toFloat()
                values[FeatureValue.ACCEL_ZERO_CROSSING_RATE.index] = magnitudes.map { it - accelMean }.zeroCrossingRate()
                values[FeatureValue.HEART_RATE_MEAN.index] = heartRates.meanOrZero()
                values[FeatureValue.HEART_RATE_STANDARD_DEVIATION.index] = heartRates.standardDeviation()
                values[FeatureValue.IBI_MEAN.index] = ibis.meanOrZero()
                values[FeatureValue.IBI_RMSSD.index] = ibis.rmssd()
                values[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] = accelValid.ratioOf(accelObserved)
                values[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] = hrValid.ratioOf(hrObserved)
                values[FeatureValue.OFF_BODY_FLAG.index] = if (epoch.any { it.quality == SensorQuality.OFF_BODY }) 1f else 0f
            }, accelValid, hrValid,
        )
    }

    private fun Extraction.isSufficient(): Boolean = values[FeatureValue.OFF_BODY_FLAG.index] == 0f &&
        values[FeatureValue.ACCEL_VALID_SAMPLE_RATIO.index] >= minimumValidSampleRatio &&
        values[FeatureValue.HEART_RATE_VALID_SAMPLE_RATIO.index] >= minimumValidSampleRatio &&
        accelValid.hasModalityCoverage(MINIMUM_ACCEL_OCCUPIED_BINS) &&
        hrValid.hasModalityCoverage(MINIMUM_HEART_RATE_OCCUPIED_BINS)

    private fun normalize(raw: FloatArray): FloatArray {
        if (priorValidRawValues.isEmpty()) return raw.copyOf()
        return raw.mapIndexed { index, value ->
            if (index == FeatureValue.OFF_BODY_FLAG.index) value else {
                val prior = priorValidRawValues.map { it[index] }
                val iqr = prior.upperQuartile() - prior.lowerQuartile()
                (value - prior.median()) / if (abs(iqr) < ZERO_IQR_EPSILON) 1f else iqr
            }
        }.toFloatArray()
    }

    private data class Extraction(val values: FloatArray, val accelValid: List<SensorSample>, val hrValid: List<SensorSample>)

    companion object {
        const val DEFAULT_EPOCH_MILLIS = 30_000L
        const val MINIMUM_VALID_SAMPLE_RATIO = 0.80f
        private const val ACTIVITY_DELTA_THRESHOLD = 0.10f
        private const val ZERO_IQR_EPSILON = 0.000001f
        /** Prototype continuous-tracker defaults: ≥24 accel bins, ≥12 HR bins, and ≥8 IBI bins per 30 seconds. */
        private const val MINIMUM_ACCEL_OCCUPIED_BINS = 24
        private const val MINIMUM_HEART_RATE_OCCUPIED_BINS = 12

        fun fromState(state: FeaturePipelineState, epochMillis: Long = DEFAULT_EPOCH_MILLIS, minimumValidSampleRatio: Float = MINIMUM_VALID_SAMPLE_RATIO): FeaturePipeline =
            FeaturePipeline(state.sessionId, epochMillis, minimumValidSampleRatio).also { pipeline ->
                pipeline.currentStartEpochMillis = state.currentStartEpochMillis
                pipeline.lastTimestampEpochMillis = state.lastTimestampEpochMillis
                pipeline.samples += state.pendingSamples
                pipeline.priorValidRawValues += state.priorValidRawValues.map(FloatArray::copyOf)
            }
    }
}

private fun List<SensorSample>.hasModalityCoverage(minimumOccupiedBins: Int): Boolean = size >= minimumOccupiedBins && occupiedSecondBins() >= minimumOccupiedBins && coverage() >= 24_000L && maxGap() <= 5_000L
private fun List<SensorSample>.coverage(): Long = last().timestampEpochMillis - first().timestampEpochMillis
private fun List<SensorSample>.maxGap(): Long = zipWithNext().maxOfOrNull { (a, b) -> b.timestampEpochMillis - a.timestampEpochMillis } ?: 0L
private fun List<SensorSample>.occupiedSecondBins(): Int = map { Math.floorDiv(it.timestampEpochMillis, 1_000L) }.distinct().size
private fun List<SensorSample>.ratioOf(observed: List<SensorSample>): Float = if (observed.isEmpty()) 0f else size.toFloat() / observed.size
private fun List<Float>.meanOrZero(): Float = if (isEmpty()) 0f else average().toFloat()
private fun List<Float>.standardDeviation(): Float = if (size < 2) 0f else sqrt(map { (it - average()) * (it - average()) }.average()).toFloat()
private fun List<Float>.median(): Float = sorted().let { if (it.isEmpty()) 0f else if (it.size % 2 == 0) (it[it.size / 2 - 1] + it[it.size / 2]) / 2f else it[it.size / 2] }
private fun List<Float>.medianAbsoluteDeviation(): Float = map { abs(it - median()) }.median()
private fun List<Float>.lowerQuartile(): Float = percentile(.25f)
private fun List<Float>.upperQuartile(): Float = percentile(.75f)
private fun List<Float>.percentile(fraction: Float): Float { if (isEmpty()) return 0f; val sorted = sorted(); val p = fraction * sorted.lastIndex; val lo = p.toInt(); val hi = ceil(p.toDouble()).toInt(); return if (lo == hi) sorted[lo] else sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo) }
private fun List<Float>.rmssd(): Float = if (size < 2) 0f else sqrt(zipWithNext().map { (a, b) -> (b - a) * (b - a) }.average()).toFloat()
private fun List<Float>.zeroCrossingRate(): Float { val nonZero = filter { it != 0f }; return if (nonZero.size < 2) 0f else nonZero.zipWithNext().count { (a, b) -> a * b < 0f }.toFloat() / (nonZero.size - 1) }
