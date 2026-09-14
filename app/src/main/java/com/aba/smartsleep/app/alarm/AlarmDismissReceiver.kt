package com.aba.smartsleep.app.alarm

import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class AlarmDismissReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_DISMISS) {
            return
        }

        val sessionId = intent.getStringExtra(EXTRA_SESSION_ID).orEmpty()
        AlarmAudio.stop()
        context.getSystemService(NotificationManager::class.java).cancel(sessionId.hashCode())
    }

    companion object {
        private const val ACTION_DISMISS = "com.aba.smartsleep.app.alarm.DISMISS"
        private const val EXTRA_SESSION_ID = "session_id"

        fun dismissIntent(context: Context, sessionId: String): Intent =
            Intent(context, AlarmDismissReceiver::class.java)
                .setAction(ACTION_DISMISS)
                .putExtra(EXTRA_SESSION_ID, sessionId)
    }
}
