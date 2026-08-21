package com.aba.smartsleep.app.alarm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.BroadcastReceiver
import android.content.Intent
import android.media.Ringtone
import android.media.RingtoneManager

class AlarmReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_FALLBACK) {
            return
        }

        val appContext = context.applicationContext
        showAlarmNotification(appContext, intent.getStringExtra(EXTRA_SESSION_ID).orEmpty())
        AlarmCoordinator(PhoneAlarmOutput(appContext)).trigger(AlarmReason.FALLBACK)
    }

    private fun showAlarmNotification(context: Context, sessionId: String) {
        val notificationManager = context.getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Alarm", NotificationManager.IMPORTANCE_HIGH),
        )
        val fullScreenIntent = PendingIntent.getBroadcast(
            context,
            sessionId.hashCode(),
            fallbackIntent(context, sessionId),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Smart Sleep alarm")
            .setContentText("Your scheduled alarm is ringing")
            .setCategory(Notification.CATEGORY_ALARM)
            .setFullScreenIntent(fullScreenIntent, true)
            .setOngoing(true)
            .build()

        notificationManager.notify(sessionId.hashCode(), notification)
    }

    companion object {
        private const val ACTION_FALLBACK = "com.aba.smartsleep.app.alarm.FALLBACK"
        private const val EXTRA_SESSION_ID = "session_id"
        private const val CHANNEL_ID = "smart_sleep_alarm"

        fun fallbackIntent(context: Context, sessionId: String): Intent =
            Intent(context, AlarmReceiver::class.java)
                .setAction(ACTION_FALLBACK)
                .putExtra(EXTRA_SESSION_ID, sessionId)
    }
}

private class PhoneAlarmOutput(context: Context) : AlarmOutput {
    private val appContext = context.applicationContext

    override fun startWatchHaptics() = Unit

    override fun startPhoneAudio() {
        activeRingtone = RingtoneManager.getRingtone(
            appContext,
            RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM),
        ).also(Ringtone::play)
    }

    private companion object {
        var activeRingtone: Ringtone? = null
    }
}
