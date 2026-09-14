package com.aba.smartsleep.wear.transport

import com.google.android.gms.wearable.DataEvent
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.WearableListenerService
import java.io.File

class WearAcknowledgementListener : WearableListenerService() {
    override fun onDataChanged(events: DataEventBuffer) {
        try {
            events.filter { it.type == DataEvent.TYPE_CHANGED }.forEach { event ->
                val path = event.dataItem.uri.path ?: return@forEach
                WatchAcknowledgementHandler.handle(
                    path,
                    PendingBatchFiles(File(filesDir, PENDING_BATCH_DIRECTORY)),
                )
            }
        } finally {
            events.release()
        }
    }

    private companion object {
        const val PENDING_BATCH_DIRECTORY = "sensor-batches"
    }
}
