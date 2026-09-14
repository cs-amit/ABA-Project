package com.aba.smartsleep.core.features

import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FeaturePipelineTest {
    @Test
    fun `emits one completed 30 second epoch`() {
        val pipeline = FeaturePipeline(epochMillis = 30_000)

        samplesFrom(0, 30_000, everyMillis = 40).flatMap { pipeline.append(it) }

        assertEquals(1, pipeline.completedEpochs.size)
    }

    @Test
    fun `does not use samples after epoch end`() {
        val epoch = FeaturePipeline().consume(samplesAroundBoundary()).single()

        assertEquals(30_000, epoch.endEpochMillis)
        assertEquals(1f, epoch.values[FeatureValue.ACCEL_MAGNITUDE_MEAN.index])
    }

    @Test
    fun `model input keeps only the nine trainable motion and heart rate values`() {
        val epoch = FeatureEpoch(
            sessionId = "s1",
            startEpochMillis = 0,
            endEpochMillis = 30_000,
            values = FloatArray(FeatureValue.entries.size) { it.toFloat() },
            validForInference = true,
        )

        assertEquals(listOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 9f, 10f), epoch.modelInput().toList())
    }

    @Test
    fun `first quality epoch seeds a baseline but is not inference ready`() {
        val epoch = FeaturePipeline().consume(combinedWindow(0, 1f)).single()

        assertFalse(epoch.validForInference)
    }

    @Test
    fun `24 accel 12 heart-rate and 8 IBI bins become inference ready after baseline`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        val epoch = pipeline.consume(mixedWindow(30_000, includeIbi = true)).single()

        assertTrue(epoch.validForInference)
    }

    @Test
    fun `missing IBI does not block motion and heart rate inference readiness`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        val epoch = pipeline.consume(mixedWindow(30_000, includeIbi = false)).single()

        assertTrue(epoch.validForInference)
    }

    @Test
    fun `trailing samples do not make an incompletely observed window inference ready`() {
        val pipeline = FeaturePipeline()

        val epoch = pipeline.consume(
            listOf(sampleAt(29_000), sampleAt(30_000)),
        ).single()

        assertFalse(epoch.validForInference)
    }

    @Test
    fun `normalizes with an actual nonzero subunit IQR from prior valid epochs`() {
        val pipeline = FeaturePipeline()
        val emitted = buildList {
            listOf(1f, 1.2f, 1.4f, 1.6f, 1.9f).forEachIndexed { index, magnitude ->
                addAll(pipeline.consume(combinedWindow(index * 30_000L, magnitude)))
            }
        }

        assertEquals(2f, emitted.last().values[FeatureValue.ACCEL_MAGNITUDE_MEAN.index], 0.0001f)
        assertTrue(emitted.last().validForInference)
    }

    @Test
    fun `sparse mixed observations do not satisfy continuous tracker quality`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        val epoch = pipeline.consume(sparseMixedWindow(30_000)).single()

        assertFalse(epoch.validForInference)
    }

    @Test
    fun `twelve observations across 24 seconds are not credible accelerometer coverage`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        assertFalse(pipeline.consume(twelvePointWindow(30_000)).single().validForInference)
    }

    @Test
    fun `degraded rows cannot supply missing accelerometer coverage bins`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        val epoch = pipeline.consume(windowWithInvalidAccelCoverage(30_000, partial = false)).single()

        assertFalse(epoch.validForInference)
    }

    @Test
    fun `partial accelerometer rows cannot supply missing coverage bins`() {
        val pipeline = FeaturePipeline()
        pipeline.consume(combinedWindow(0, 1f))

        val epoch = pipeline.consume(windowWithInvalidAccelCoverage(30_000, partial = true)).single()

        assertFalse(epoch.validForInference)
    }

    private fun samplesFrom(start: Long, endInclusive: Long, everyMillis: Long): List<SensorSample> =
        generateSequence(start) { timestamp -> (timestamp + everyMillis).takeIf { it <= endInclusive } }
            .map { sampleAt(it) }
            .toList()

    private fun samplesAroundBoundary(): List<SensorSample> = listOf(
        sampleAt(0, accelX = 1f),
        sampleAt(10_000, accelX = 1f),
        sampleAt(20_000, accelX = 1f),
        sampleAt(29_999, accelX = 1f),
        sampleAt(30_000, accelX = 100f),
    )

    private fun sampleAt(timestamp: Long, accelX: Float = 1f, ibiMillis: Int? = 1_000) = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = accelX,
        accelY = 0f,
        accelZ = 0f,
        heartRateBpm = 60f,
        ibiMillis = ibiMillis,
        quality = SensorQuality.VALID,
    )

    private fun combinedWindow(start: Long, accelX: Float): List<SensorSample> =
        samplesFrom(start, start + 30_000, everyMillis = 1_000).map { sample ->
            sample.copy(accelX = accelX)
        }

    private fun mixedWindow(start: Long, includeIbi: Boolean): List<SensorSample> = buildList {
        ((0L..22_000L step 1_000L) + 24_000L).forEach { offset ->
            add(accelOnlyAt(start + offset))
        }
        val heartRateOffsets = listOf(0L, 2_000L, 4_000L, 6_000L, 8_000L, 10_000L, 12_000L, 14_000L, 16_000L, 18_000L, 20_000L, 24_000L)
        val ibiOffsets = setOf(0L, 4_000L, 8_000L, 12_000L, 16_000L, 18_000L, 20_000L, 24_000L)
        heartRateOffsets.forEach { offset ->
            add(
                SensorSample(
                    timestampEpochMillis = start + offset,
                    accelX = null,
                    accelY = null,
                    accelZ = null,
                    heartRateBpm = 60f,
                    ibiMillis = if (includeIbi && offset in ibiOffsets) 1_000 else null,
                    quality = SensorQuality.VALID,
                ),
            )
        }
        add(boundaryAt(start + 30_000))
    }.sortedBy(SensorSample::timestampEpochMillis)

    private fun sparseMixedWindow(start: Long): List<SensorSample> = buildList {
        (start..start + 25_000 step 5_000).forEach { timestamp ->
            add(sampleAt(timestamp, ibiMillis = null))
            add(SensorSample(timestamp, null, null, null, 60f, 1_000, SensorQuality.VALID))
        }
        add(sampleAt(start + 30_000))
    }.sortedBy(SensorSample::timestampEpochMillis)

    private fun twelvePointWindow(start: Long): List<SensorSample> = buildList {
        listOf(0L, 2_000L, 4_000L, 6_000L, 8_000L, 10_000L, 12_000L, 14_000L, 16_000L, 18_000L, 20_000L, 24_000L).forEach { offset ->
            add(sampleAt(start + offset, ibiMillis = null))
            add(SensorSample(start + offset, null, null, null, 60f, 1_000, SensorQuality.VALID))
        }
        add(sampleAt(start + 30_000))
    }.sortedBy(SensorSample::timestampEpochMillis)

    private fun windowWithInvalidAccelCoverage(start: Long, partial: Boolean): List<SensorSample> = buildList {
        (0L..18_000L step 1_000L).forEach { offset -> add(accelOnlyAt(start + offset)) }
        (0L..4_000L step 1_000L).forEach { offset -> add(accelOnlyAt(start + offset + 100)) }
        (19_000L..24_000L step 1_000L).forEach { offset ->
            add(
                SensorSample(
                    timestampEpochMillis = start + offset,
                    accelX = 1f,
                    accelY = if (partial) null else 0f,
                    accelZ = if (partial) null else 0f,
                    heartRateBpm = null,
                    ibiMillis = null,
                    quality = if (partial) SensorQuality.VALID else SensorQuality.DEGRADED,
                ),
            )
        }
        addAll(mixedWindow(start, includeIbi = true).filter { it.accelX == null })
    }.sortedBy(SensorSample::timestampEpochMillis)

    private fun accelOnlyAt(timestamp: Long) = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = 1f,
        accelY = 0f,
        accelZ = 0f,
        heartRateBpm = null,
        ibiMillis = null,
        quality = SensorQuality.VALID,
    )

    private fun boundaryAt(timestamp: Long) = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = null,
        accelY = null,
        accelZ = null,
        heartRateBpm = null,
        ibiMillis = null,
        quality = SensorQuality.DEGRADED,
    )
}
