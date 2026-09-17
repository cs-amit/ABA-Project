package com.aba.smartsleep.wear.sensor

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class ActiveTrackerFlusherTest {
    @Test
    fun `flushes once per interval and stops with capture`() = runBlocking {
        val ticks = Channel<Unit>(Channel.UNLIMITED)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
        var flushCount = 0
        val flusher = ActiveTrackerFlusher(scope, { flushCount++ }) { interval ->
            assertEquals(5_000L, interval)
            ticks.receive()
        }
        try {
            flusher.start()
            flusher.start() // Repeated capture starts must not multiply flush requests.
            assertEquals(0, flushCount)
            ticks.send(Unit)
            assertEquals(1, flushCount)
            ticks.send(Unit)
            assertEquals(2, flushCount)
            flusher.stop()
            ticks.send(Unit)
            assertEquals(2, flushCount)
        } finally {
            scope.cancel()
        }
    }

    @Test
    fun `can restart after stopping a capture`() = runBlocking {
        val ticks = Channel<Unit>(Channel.UNLIMITED)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
        var flushCount = 0
        val flusher = ActiveTrackerFlusher(scope, { flushCount++ }) { ticks.receive() }
        try {
            flusher.start()
            ticks.send(Unit)
            flusher.stop()
            flusher.start()
            ticks.send(Unit)
            assertEquals(2, flushCount)
        } finally {
            scope.cancel()
        }
    }
}
