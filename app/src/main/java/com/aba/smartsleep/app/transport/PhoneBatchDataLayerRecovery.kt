package com.aba.smartsleep.app.transport

import android.content.Context
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.PutDataRequest
import com.google.android.gms.wearable.Wearable
import java.nio.ByteBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

internal object PhoneBatchDataLayerRecovery {
    fun recover(context: Context) {
        val dataClient = Wearable.getDataClient(context)
        val dataItems = Tasks.await(dataClient.dataItems)
        try {
            val persistedBatches = dataItems.mapNotNull { item ->
                val path = item.uri.path ?: return@mapNotNull null
                val payload = item.data ?: return@mapNotNull null
                PersistedBatchDataItem(path, payload.copyOf())
            }
            WearableReceiver.recover(persistedBatches)
        } finally {
            dataItems.release()
        }
    }

    fun acknowledge(context: Context, decision: PhoneDataLayerHandler.AcknowledgementDecision): Boolean =
        runCatching {
            Tasks.await(Wearable.getDataClient(context).putDataItem(acknowledgementRequest(decision)))
            true
        }.getOrDefault(false)

    fun acknowledge(context: Context, key: PhoneBatchInbox.DeliveryKey): Boolean =
        acknowledge(context, PhoneDataLayerHandler.AcknowledgementDecision(key.sessionId, key.sequence))

    private fun acknowledgementRequest(
        decision: PhoneDataLayerHandler.AcknowledgementDecision,
    ): PutDataRequest = PutDataRequest.create(
        "/sessions/${decision.sessionId}/acknowledgements/${decision.sequence}",
    )
        .setData(ByteBuffer.allocate(Long.SIZE_BYTES).putLong(decision.sequence).array())
        .setUrgent()
}

internal class WatchAcknowledgementRetryCoordinator(
    private val pendingKeys: suspend () -> List<PhoneBatchInbox.DeliveryKey>,
    private val send: suspend (PhoneBatchInbox.DeliveryKey) -> Boolean,
    private val markDelivered: suspend (PhoneBatchInbox.DeliveryKey) -> Unit,
    private val initialBackoffMillis: Long = INITIAL_BACKOFF_MILLIS,
    private val maximumBackoffMillis: Long = MAX_BACKOFF_MILLIS,
) {
    init {
        require(initialBackoffMillis > 0)
        require(maximumBackoffMillis >= initialBackoffMillis)
    }

    private val workerMutex = Mutex()
    private val wakeSignals = Channel<Unit>(Channel.CONFLATED)
    private var retryJob: Job? = null

    /** Starts or wakes the sole process-lifetime worker. Only this worker sends or deletes keys. */
    suspend fun ensureRetryLoop(scope: CoroutineScope): Job = workerMutex.withLock {
        retryJob?.takeIf(Job::isActive)?.let { activeWorker ->
            wakeSignals.trySend(Unit)
            return@withLock activeWorker
        }
        scope.launch {
            var delayMillis = initialBackoffMillis
            while (isActive) {
                val hasPending = processPendingOnce()
                if (!hasPending) {
                    delayMillis = initialBackoffMillis
                    wakeSignals.receive()
                } else {
                    withTimeoutOrNull(delayMillis) { wakeSignals.receive() }
                    delayMillis = (delayMillis * 2).coerceAtMost(maximumBackoffMillis)
                }
            }
        }.also { retryJob = it }
    }

    private suspend fun processPendingOnce(): Boolean {
        val keys = try {
            pendingKeys()
        } catch (error: CancellationException) {
            throw error
        } catch (_: Throwable) {
            return true
        }
        keys.forEach { key ->
            val sent = try {
                send(key)
            } catch (error: CancellationException) {
                throw error
            } catch (_: Throwable) {
                false
            }
            if (sent) {
                try {
                    markDelivered(key)
                } catch (error: CancellationException) {
                    throw error
                } catch (_: Throwable) {
                    // The durable key remains pending and will be retried.
                }
            }
        }
        return try {
            pendingKeys().isNotEmpty()
        } catch (error: CancellationException) {
            throw error
        } catch (_: Throwable) {
            true
        }
    }

    private companion object {
        const val INITIAL_BACKOFF_MILLIS = 1_000L
        const val MAX_BACKOFF_MILLIS = 60_000L
    }
}
