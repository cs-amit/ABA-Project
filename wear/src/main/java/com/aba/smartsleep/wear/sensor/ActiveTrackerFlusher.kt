package com.aba.smartsleep.wear.sensor

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/** Prevents screen-off SDK batching from delaying live heart-rate data for minutes. */
internal class ActiveTrackerFlusher(
    private val scope: CoroutineScope,
    private val flush: () -> Unit,
    private val waitForInterval: suspend (Long) -> Unit = { delay(it) },
) {
    private var job: Job? = null

    fun start() {
        if (job?.isActive == true) return
        job = scope.launch {
            while (isActive) {
                waitForInterval(5_000L)
                if (isActive) flush()
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}
