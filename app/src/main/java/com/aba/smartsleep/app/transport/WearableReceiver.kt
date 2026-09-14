package com.aba.smartsleep.app.transport

import kotlinx.coroutines.flow.Flow
import java.io.File

object WearableReceiver {
    private var coordinator: PhoneBatchReleaseCoordinator? = null
    private var handler: PhoneDataLayerHandler? = null

    val receivedBatches: Flow<PhoneBatchDelivery>
        get() = synchronized(this) {
            requireNotNull(coordinator) { "WearableReceiver must be initialized before receiving batches." }
                .deliveries()
        }

    fun initialize(filesDirectory: File) {
        synchronized(this) {
            if (coordinator == null) {
                coordinator = PhoneBatchReleaseCoordinator(PhoneBatchInbox(File(filesDirectory, INBOX_DIRECTORY)))
                handler = PhoneDataLayerHandler(requireNotNull(coordinator))
            }
        }
    }

    fun accept(path: String, payload: ByteArray): PhoneDataLayerHandler.AcknowledgementDecision? = synchronized(this) {
        requireNotNull(handler) {
            "WearableReceiver must be initialized before receiving batches."
        }.handle(path, payload)
    }

    fun recover(items: Iterable<PersistedBatchDataItem>): List<PhoneDataLayerHandler.AcknowledgementDecision> =
        synchronized(this) {
            PhoneBatchStartupReplay(
                requireNotNull(handler) {
                    "WearableReceiver must be initialized before recovering batches."
                },
            ).recover(items)
        }

    private const val INBOX_DIRECTORY = "received-sensor-batches"
}
