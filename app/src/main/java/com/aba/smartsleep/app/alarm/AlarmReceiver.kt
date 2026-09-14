package com.aba.smartsleep.app.alarm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.Manifest
import android.os.Build
import android.media.Ringtone
import android.media.RingtoneManager

class AlarmReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_FALLBACK) {
            return
        }

        val appContext = context.applicationContext
        val sessionId = intent.getStringExtra(EXTRA_SESSION_ID).orEmpty()
        AlarmCoordinator(sessionId, PhoneAlarmOutput(appContext)).trigger(AlarmReason.FALLBACK)
        if (canPostNotifications(appContext)) {
            showAlarmNotification(appContext, sessionId)
        }
    }

    private fun showAlarmNotification(context: Context, sessionId: String) {
        val notificationManager = context.getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Alarm", NotificationManager.IMPORTANCE_HIGH),
        )
        val fullScreenIntent = PendingIntent.getActivity(
            context,
            sessionId.hashCode() xor DISPLAY_REQUEST_CODE_MASK,
            AlarmActivity.displayIntent(context, sessionId),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val dismissIntent = PendingIntent.getBroadcast(
            context,
            sessionId.hashCode() xor DISMISS_REQUEST_CODE_MASK,
            AlarmDismissReceiver.dismissIntent(context, sessionId),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Smart Sleep alarm")
            .setContentText("Your scheduled alarm is ringing")
            .setCategory(Notification.CATEGORY_ALARM)
            .setContentIntent(fullScreenIntent)
            .setFullScreenIntent(fullScreenIntent, true)
            .addAction(Notification.Action.Builder(0, "Dismiss", dismissIntent).build())
            .setOngoing(true)
            .build()

        notificationManager.notify(sessionId.hashCode(), notification)
    }

    private fun canPostNotifications(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

    companion object {
        private const val ACTION_FALLBACK = "com.aba.smartsleep.app.alarm.FALLBACK"
        private const val EXTRA_SESSION_ID = "session_id"
        private const val CHANNEL_ID = "smart_sleep_alarm"
        private const val DISPLAY_REQUEST_CODE_MASK = 0x51A7
        private const val DISMISS_REQUEST_CODE_MASK = 0xD155

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
        AlarmAudio.start(appContext)
    }
}

internal object AlarmAudio {
    private var activeRingtone: Ringtone? = null

    fun start(context: Context) {
        activeRingtone = RingtoneManager.getRingtone(
            context,
            RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM),
        ).also(Ringtone::play)
    }

    fun stop() {
        activeRingtone?.stop()
        activeRingtone = null
    }
}
