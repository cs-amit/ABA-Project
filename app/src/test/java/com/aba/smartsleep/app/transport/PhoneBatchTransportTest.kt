package com.aba.smartsleep.app.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.transport.SensorBatchCodec
import java.nio.file.Files
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneBatchTransportTest {
    @Test
    fun `listener handler acknowledges only after its content-synced journal accepts the payload`() {
        val root = Files.createTempDirectory("phone-batch-inbox").toFile()
        val batch = batch("s1", 1_000)
        val payload = SensorBatchCodec.encode(batch)
        val failingHandler = PhoneDataLayerHandler(
            PhoneBatchReleaseCoordinator(
                PhoneBatchInbox(root, AtomicPayloadWriter { target, _ ->
                    target.writeBytes(byteArrayOf(99))
                    false
                }, JournalRecordWriter { _, _ -> false }),
            ),
        )

        val failedDecision = failingHandler.handle(
            "/sessions/s1/batches/0",
            payload,
        )
        val retryCoordinator = PhoneBatchReleaseCoordinator(
            PhoneBatchInbox(root, AtomicPayloadWriter { target, _ ->
                target.writeBytes(byteArrayOf(99))
                false
            }),
        )
        val retryDecision = PhoneDataLayerHandler(retryCoordinator).handle(
            "/sessions/s1/batches/0",
            payload,
        )

        assertNull(failedDecision)
        assertTrue(retryDecision!!.shouldAcknowledge)
        assertEquals(batch, runBlocking { withTimeout(1_000) { retryCoordinator.deliveries().first().batch } })
    }

    @Test
    fun `arrival during subscription is delivered from the synchronized persisted release queue`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-order").toFile()
        val coordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        val handler = PhoneDataLayerHandler(coordinator)
        val first = batch("s1", 1_000)

        val consumer = async(start = CoroutineStart.UNDISPATCHED) {
            coordinator.deliveries().first().batch
        }
        handler.handle("/sessions/s1/batches/0", SensorBatchCodec.encode(first))

        assertEquals(first, withTimeout(1_000) { consumer.await() })
    }

    @Test
    fun `out of order arrivals are released once in sequence and stay released after collector restart`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-rejection").toFile()
        val coordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        val handler = PhoneDataLayerHandler(coordinator)
        val first = batch("s1", 1_000)
        val second = batch("s1", 2_000)

        handler.handle("/sessions/s1/batches/1", SensorBatchCodec.encode(second))
        val consumer = async(start = CoroutineStart.UNDISPATCHED) {
            buildList {
                coordinator.deliveries().take(2).collect { delivery ->
                    add(delivery.batch)
                    assertTrue(delivery.confirmPersisted())
                }
            }
        }
        handler.handle("/sessions/s1/batches/0", SensorBatchCodec.encode(first))

        assertEquals(listOf(first, second), withTimeout(1_000) { consumer.await() })
        assertNull(withTimeoutOrNull(100) { coordinator.deliveries().first() })
    }

    @Test
    fun `interrupted release lease resumes on restart until consumer confirmation`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-interrupted-release").toFile()
        val inbox = PhoneBatchInbox(root)
        val first = batch("s1", 1_000)
        assertTrue(PhoneDataLayerHandler(PhoneBatchReleaseCoordinator(inbox)).handle(
            "/sessions/s1/batches/0",
            SensorBatchCodec.encode(first),
        )!!.shouldAcknowledge)

        assertEquals(first, inbox.beginReleaseNext()!!.batch)

        val restarted = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        val retried = withTimeout(1_000) { restarted.deliveries().first() }
        assertEquals(first, retried.batch)
        assertTrue(retried.confirmPersisted())
        assertNull(withTimeoutOrNull(100) {
            PhoneBatchReleaseCoordinator(PhoneBatchInbox(root)).deliveries().first()
        })
    }

    @Test
    fun `startup Data Item replay recovers an acknowledged batch when the new journal entry is lost`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-lost-journal-entry").toFile()
        val first = batch("s1", 1_000)
        val payload = SensorBatchCodec.encode(first)
        val interruptedHandler = PhoneDataLayerHandler(
            PhoneBatchReleaseCoordinator(
                PhoneBatchInbox(
                    root,
                    AtomicPayloadWriter { _, _ -> false },
                    JournalRecordWriter { journal, record ->
                        if (!FileContentSyncedJournalWriter.append(journal, record)) {
                            false
                        } else {
                            check(journal.delete()) { "The test must simulate loss of the new journal entry." }
                            true
                        }
                    },
                ),
            ),
        )

        assertTrue(interruptedHandler.handle("/sessions/s1/batches/0", payload)!!.shouldAcknowledge)

        val restartedCoordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        val recovered = PhoneBatchStartupReplay(
            PhoneDataLayerHandler(restartedCoordinator),
        ).recover(listOf(PersistedBatchDataItem("/sessions/s1/batches/0", payload)))

        assertEquals(1, recovered.size)
        assertTrue(recovered.single().shouldAcknowledge)
        assertEquals(first, withTimeout(1_000) { restartedCoordinator.deliveries().first().batch })
    }

    @Test
    fun `emitted batch is retried after restart until the consumer confirms its transaction`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-explicit-completion").toFile()
        val first = batch("s1", 1_000)
        val payload = SensorBatchCodec.encode(first)
        val coordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        assertTrue(PhoneDataLayerHandler(coordinator).handle(
            "/sessions/s1/batches/0",
            payload,
        )!!.shouldAcknowledge)

        val interruptedDelivery = withTimeout(1_000) { coordinator.deliveries().first() }
        assertEquals(first, interruptedDelivery.batch)

        val restartedCoordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        val retriedDelivery = withTimeout(1_000) { restartedCoordinator.deliveries().first() }
        assertEquals(PhoneBatchInbox.DeliveryKey("s1", 0), retriedDelivery.key)
        assertEquals(first, retriedDelivery.batch)
        assertTrue(retriedDelivery.confirmPersisted())

        val afterCompletion = PhoneBatchReleaseCoordinator(PhoneBatchInbox(root))
        assertNull(withTimeoutOrNull(100) { afterCompletion.deliveries().first() })
    }

    @Test
    fun `false confirmation is retried without blocking the live ordered flow`() = runBlocking {
        val root = Files.createTempDirectory("phone-batch-live-retry").toFile()
        var rejectFirstCompletion = true
        val coordinator = PhoneBatchReleaseCoordinator(
            PhoneBatchInbox(root, journalWriter = JournalRecordWriter { journal, record ->
                if (record.startsWith("C\t") && rejectFirstCompletion) {
                    rejectFirstCompletion = false
                    false
                } else {
                    FileContentSyncedJournalWriter.append(journal, record)
                }
            }),
        )
        val first = batch("s1", 1_000)
        assertTrue(PhoneDataLayerHandler(coordinator).handle(
            "/sessions/s1/batches/0",
            SensorBatchCodec.encode(first),
        )!!.shouldAcknowledge)

        val attempts = withTimeout(1_000) {
            buildList {
                coordinator.deliveries().take(2).collect { delivery ->
                    add(delivery)
                    if (size == 1) assertTrue(!delivery.confirmPersisted())
                    else assertTrue(delivery.confirmPersisted())
                }
            }
        }

        assertEquals(listOf(0L, 0L), attempts.map { it.key.sequence })
    }

    @Test
    fun `unsafe path segments and mismatched sessions are rejected before acknowledgement`() {
        val root = Files.createTempDirectory("phone-batch-unsafe-session").toFile()
        val handler = PhoneDataLayerHandler(PhoneBatchReleaseCoordinator(PhoneBatchInbox(root)))
        val otherSessionPayload = SensorBatchCodec.encode(batch("s2", 1_000))

        assertNull(handler.handle("/sessions/../batches/0", SensorBatchCodec.encode(batch("..", 1_000))))
        assertNull(handler.handle("/sessions/./batches/0", SensorBatchCodec.encode(batch(".", 1_000))))
        assertNull(handler.handle("/sessions/s1/batches/not-a-sequence", otherSessionPayload))
        assertNull(handler.handle("/sessions/s1/batches/0", otherSessionPayload))
    }

    private fun batch(sessionId: String, timestamp: Long): SensorBatch = SensorBatch(
        sessionId,
        listOf(
            SensorSample(
                timestampEpochMillis = timestamp,
                accelX = null,
                accelY = null,
                accelZ = null,
                heartRateBpm = 60f,
                ibiMillis = null,
                quality = SensorQuality.VALID,
            ),
        ),
    )
}
