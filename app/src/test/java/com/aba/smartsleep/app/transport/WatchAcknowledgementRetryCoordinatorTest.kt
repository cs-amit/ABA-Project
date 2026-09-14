package com.aba.smartsleep.app.transport

import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WatchAcknowledgementRetryCoordinatorTest {
    @Test
    fun `concurrent wakeups share one worker and serialize acknowledgement sends`() = runBlocking {
        val key = PhoneBatchInbox.DeliveryKey("session-1", 4)
        val pending = Collections.synchronizedList(mutableListOf(key))
        val enteredSend = CompletableDeferred<Unit>()
        val releaseSend = CompletableDeferred<Unit>()
        val activeSends = AtomicInteger()
        val maximumActiveSends = AtomicInteger()
        val coordinator = WatchAcknowledgementRetryCoordinator(
            pendingKeys = { pending.toList() },
            send = {
                val active = activeSends.incrementAndGet()
                maximumActiveSends.updateAndGet { previous -> maxOf(previous, active) }
                enteredSend.complete(Unit)
                releaseSend.await()
                activeSends.decrementAndGet()
                true
            },
            markDelivered = { pending.remove(it) },
        )
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

        val jobs = List(32) { async(Dispatchers.Default) { coordinator.ensureRetryLoop(scope) } }.awaitAll()
        enteredSend.await()
        repeat(10) { yield() }

        assertEquals(1, jobs.toSet().size)
        assertEquals(1, maximumActiveSends.get())
        releaseSend.complete(Unit)
        awaitPendingEmpty(pending)
        jobs.first().cancelAndJoin()
        scope.cancel()
    }

    @Test
    fun `persisted key wakes worker immediately while it is backing off`() = runBlocking {
        val key = PhoneBatchInbox.DeliveryKey("session-1", 5)
        val pending = mutableListOf(key)
        val firstFailure = CompletableDeferred<Unit>()
        var attempts = 0
        val coordinator = WatchAcknowledgementRetryCoordinator(
            pendingKeys = { pending.toList() },
            send = {
                attempts += 1
                (attempts > 1).also { if (!it) firstFailure.complete(Unit) }
            },
            markDelivered = { pending.remove(it) },
        )

        val worker = coordinator.ensureRetryLoop(this)
        firstFailure.await()
        coordinator.ensureRetryLoop(this)

        withTimeout(500) { awaitPendingEmpty(pending) }
        assertEquals(2, attempts)
        worker.cancelAndJoin()
    }

    @Test
    fun `persisted key wakes an idle worker`() = runBlocking {
        val key = PhoneBatchInbox.DeliveryKey("session-1", 8)
        val pending = mutableListOf<PhoneBatchInbox.DeliveryKey>()
        var pendingReads = 0
        val workerBecameIdle = CompletableDeferred<Unit>()
        val coordinator = WatchAcknowledgementRetryCoordinator(
            pendingKeys = {
                pendingReads += 1
                if (pendingReads >= 2) workerBecameIdle.complete(Unit)
                pending.toList()
            },
            send = { true },
            markDelivered = { pending.remove(it) },
        )

        val worker = coordinator.ensureRetryLoop(this)
        workerBecameIdle.await()
        pending += key
        coordinator.ensureRetryLoop(this)

        withTimeout(500) { awaitPendingEmpty(pending) }
        worker.cancelAndJoin()
    }

    @Test
    fun `worker recovers from pending send and delete failures until durable key is removed`() = runBlocking {
        val key = PhoneBatchInbox.DeliveryKey("session-1", 6)
        val pending = mutableListOf(key)
        var pendingReads = 0
        var sendAttempts = 0
        var deleteAttempts = 0
        val coordinator = WatchAcknowledgementRetryCoordinator(
            pendingKeys = {
                pendingReads += 1
                if (pendingReads == 1) error("temporary DAO read failure")
                pending.toList()
            },
            send = {
                sendAttempts += 1
                when (sendAttempts) {
                    1 -> error("temporary Data Layer failure")
                    2 -> false
                    else -> true
                }
            },
            markDelivered = {
                deleteAttempts += 1
                if (deleteAttempts == 1) error("temporary DAO delete failure")
                pending.remove(it)
            },
            initialBackoffMillis = 1,
            maximumBackoffMillis = 4,
        )

        val worker = coordinator.ensureRetryLoop(this)
        repeat(5) {
            coordinator.ensureRetryLoop(this)
            yield()
        }

        withTimeout(1_000) { awaitPendingEmpty(pending) }
        assertTrue(pendingReads >= 2)
        assertEquals(4, sendAttempts)
        assertEquals(2, deleteAttempts)
        worker.cancelAndJoin()
    }

    @Test
    fun `cancelled worker stops and a later wake starts a replacement`() = runBlocking {
        val key = PhoneBatchInbox.DeliveryKey("session-1", 7)
        val pending = mutableListOf(key)
        val firstSendStarted = CompletableDeferred<Unit>()
        var sends = 0
        val coordinator = WatchAcknowledgementRetryCoordinator(
            pendingKeys = { pending.toList() },
            send = {
                sends += 1
                if (sends == 1) {
                    firstSendStarted.complete(Unit)
                    throw CancellationException("worker cancelled during send")
                }
                true
            },
            markDelivered = { pending.remove(it) },
        )

        val first = coordinator.ensureRetryLoop(this)
        firstSendStarted.await()
        first.cancelAndJoin()
        val replacement = coordinator.ensureRetryLoop(this)

        withTimeout(500) { awaitPendingEmpty(pending) }
        assertNotEquals(first, replacement)
        assertEquals(2, sends)
        replacement.cancelAndJoin()
    }

    private suspend fun awaitPendingEmpty(pending: List<PhoneBatchInbox.DeliveryKey>) {
        while (pending.isNotEmpty()) yield()
    }
}
