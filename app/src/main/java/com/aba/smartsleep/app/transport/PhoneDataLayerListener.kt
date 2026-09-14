package com.aba.smartsleep.app.transport

import com.google.android.gms.wearable.DataEvent
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.WearableListenerService

class PhoneDataLayerListener : WearableListenerService() {
    override fun onCreate() {
        super.onCreate()
        WearableReceiver.initialize(filesDir)
    }

    override fun onDataChanged(events: DataEventBuffer) {
        try {
            events.filter { it.type == DataEvent.TYPE_CHANGED }.forEach { event ->
                val path = event.dataItem.uri.path ?: return@forEach
                val payload = event.dataItem.data ?: return@forEach
                WearableReceiver.accept(path, payload)
            }
        } finally {
            events.release()
        }
    }
}
