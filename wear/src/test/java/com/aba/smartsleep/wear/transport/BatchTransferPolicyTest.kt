package com.aba.smartsleep.wear.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.transport.SensorBatchCodec
import com.aba.smartsleep.core.features.FeaturePipeline
import java.nio.file.Files
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class BatchTransferPolicyTest {
    @Test
    fun `dense samples split before the codec payload limit`() {
        val accumulator = SensorBatchAccumulator("s1")
        val batches = buildList {
            repeat(3_000) { index -> addAll(accumulator.append(denseSample(index.toLong()))) }
            accumulator.flush()?.let(::add)
        }

        assertTrue(batches.size > 1)
        assertTrue(batches.all { SensorBatchCodec.encode(it).size < 90 * 1024 })
        assertEquals(3_000, batches.sumOf { it.samples.size })
    }

    @Test
    fun `failed transfer does not prevent later batches from being attempted`() {
        val attempted = mutableListOf<SensorBatch>()
        val first = SensorBatch("s1", listOf(denseSample(1)))
        val second = SensorBatch("s1", listOf(denseSample(2)))
        val transfer = ResilientBatchTransfer { batch ->
            attempted += batch
            if (batch == first) error("temporary upload failure")
        }

        val failures = runBlocking { transfer.transfer(listOf(first, second)) }

        assertEquals(listOf(first, second), attempted)
        assertEquals(1, failures.size)
    }

    @Test
    fun `persisted acknowledgement is reconciled after a sender restart`() {
        val root = Files.createTempDirectory("watch-batch-pending").toFile()
        val store = PendingBatchFiles(root)
        store.write("s1", 4, byteArrayOf(1, 2, 3))
        store.acknowledge("s1", 4)

        val restartedStore = PendingBatchFiles(root)

        assertTrue(restartedStore.isAcknowledged("s1", 4))
        assertEquals(emptyList<Long>(), restartedStore.pendingSequences("s1"))
        assertFalse(restartedStore.pendingFile("s1", 4).exists())
    }

    @Test
    fun `unsafe watch session segments are rejected`() {
        val store = PendingBatchFiles(Files.createTempDirectory("watch-batch-unsafe-session").toFile())

        try {
            store.write("..", 0, byteArrayOf(1))
            fail("Expected an unsafe session ID to be rejected.")
        } catch (_: IllegalArgumentException) {
        }

        try {
            store.isAcknowledged("..", 0)
            fail("Expected unsafe session queries to be rejected.")
        } catch (_: IllegalArgumentException) {
        }
    }

    @Test
    fun `cancellation is rethrown instead of being counted as a transfer failure`() {
        val batch = SensorBatch("s1", listOf(denseSample(1)))
        val transfer = ResilientBatchTransfer { throw CancellationException("capture stopped") }

        try {
            runBlocking { transfer.transfer(listOf(batch)) }
            fail("Expected cancellation to propagate.")
        } catch (_: CancellationException) {
        }
    }

    @Test
    fun `pending resend rethrows cancellation`() {
        val store = PendingBatchFiles(Files.createTempDirectory("watch-batch-cancel-resend").toFile())
        store.write("s1", 0, byteArrayOf(1))
        val resender = PendingBatchResender(store) { _, _, _ ->
            throw CancellationException("capture stopped")
        }

        try {
            runBlocking { resender.resend("s1") }
            fail("Expected resend cancellation to propagate.")
        } catch (_: CancellationException) {
        }
    }

    @Test
    fun `malformed acknowledgement path is ignored without touching watch storage`() {
        val store = PendingBatchFiles(Files.createTempDirectory("watch-batch-malformed-ack").toFile())

        assertFalse(WatchAcknowledgementHandler.handle("/sessions/../acknowledgements/0", store))
        assertFalse(WatchAcknowledgementHandler.handle("/sessions/./acknowledgements/0", store))
    }

    @Test
    fun `late callback in following accumulator batch is dropped without blocking later data`() {
        val accumulator = SensorBatchAccumulator("s1")
        accumulator.append(denseSample(0))
        accumulator.append(denseSample(30_000))
        val first = accumulator.append(denseSample(30_040)).single()
        accumulator.append(denseSample(29_980))
        val second = accumulator.flush()!!
        val pipeline = FeaturePipeline(sessionId = "s1")

        pipeline.consume(first.samples)
        pipeline.consume(second.samples)

        assertEquals(1, pipeline.completedEpochs.size)
        assertEquals(1, pipeline.droppedFinalizedSampleCount)
    }

    private fun denseSample(timestamp: Long): SensorSample = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = 1f,
        accelY = 2f,
        accelZ = 3f,
        heartRateBpm = 60f,
        ibiMillis = 800,
        quality = SensorQuality.VALID,
    )
}
