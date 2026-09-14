package com.aba.smartsleep.wear.transport

import android.content.Context
import android.util.Log
import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.transport.SensorBatchCodec
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.DataClient
import com.google.android.gms.wearable.DataEvent
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.PutDataRequest
import com.google.android.gms.wearable.Wearable
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File

class WearBatchSender(context: Context) {
    private val appContext = context.applicationContext
    private val dataClient = Wearable.getDataClient(appContext)
    private val sequenceStore = appContext.getSharedPreferences(SEQUENCE_STORE, Context.MODE_PRIVATE)
    private val pendingFiles = PendingBatchFiles(File(appContext.filesDir, PENDING_BATCH_DIRECTORY))
    private var accumulator: SensorBatchAccumulator? = null

    private val acknowledgementListener = DataClient.OnDataChangedListener(::handleAcknowledgements)

    init {
        dataClient.addListener(acknowledgementListener)
    }

    suspend fun collect(sessionId: String, samples: Flow<SensorSample>) {
        validateSessionId(sessionId)
        accumulator = SensorBatchAccumulator(sessionId)
        try {
            resendPending(sessionId)
        } catch (error: CancellationException) {
            throw error
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to resend pending sensor batches.", error)
        }
        samples.collect { sample ->
            val completed = synchronized(this) {
                requireNotNull(accumulator).append(sample)
            }
            transfer(completed)
        }
    }

    suspend fun send(batch: SensorBatch) {
        validateSessionId(batch.sessionId)
        val payload = SensorBatchCodec.encode(batch)
        val sequence = nextSequence(batch.sessionId)
        pendingFiles.write(batch.sessionId, sequence, payload)
        upload(batch.sessionId, sequence, payload)
    }

    suspend fun flush() {
        val batch = synchronized(this) { accumulator?.flush().also { accumulator = null } }
        batch?.let { transfer(listOf(it)) }
    }

    fun close() {
        dataClient.removeListener(acknowledgementListener)
    }

    private suspend fun resendPending(sessionId: String) {
        PendingBatchResender(pendingFiles, ::upload).resend(sessionId).forEach { error ->
            Log.w(TAG, "Unable to resend pending sensor batch.", error)
        }
    }

    private suspend fun transfer(batches: List<SensorBatch>) {
        ResilientBatchTransfer(::send).transfer(batches).forEach { error ->
            Log.w(TAG, "Sensor batch transfer failed; capture will continue.", error)
        }
    }

    private suspend fun upload(sessionId: String, sequence: Long, payload: ByteArray) {
        val request = PutDataRequest.create(batchPath(sessionId, sequence))
            .setData(payload)
            .setUrgent()
        withContext(Dispatchers.IO) {
            Tasks.await(dataClient.putDataItem(request))
        }
    }

    private fun nextSequence(sessionId: String): Long = synchronized(sequenceStore) {
        val key = "$SEQUENCE_KEY_PREFIX$sessionId"
        val sequence = sequenceStore.getLong(key, 0L)
        check(sequence >= 0) { "Invalid stored batch sequence." }
        check(sequence < Long.MAX_VALUE) { "Batch sequence is exhausted." }
        check(sequenceStore.edit().putLong(key, sequence + 1).commit()) {
            "Unable to persist the next batch sequence."
        }
        sequence
    }

    private fun handleAcknowledgements(events: DataEventBuffer) {
        try {
            events.filter { it.type == DataEvent.TYPE_CHANGED }.forEach { event ->
                event.dataItem.uri.path?.let { path -> WatchAcknowledgementHandler.handle(path, pendingFiles) }
            }
        } finally {
            events.release()
        }
    }

    private fun validateSessionId(sessionId: String) {
        requireSafeWatchSessionId(sessionId)
    }

    private fun batchPath(sessionId: String, sequence: Long): String =
        "/sessions/$sessionId/batches/$sequence"

    private companion object {
        const val PENDING_BATCH_DIRECTORY = "sensor-batches"
        const val SEQUENCE_STORE = "wear-batch-sequences"
        const val SEQUENCE_KEY_PREFIX = "next-sequence-"
        const val TAG = "WearBatchSender"
    }
}
