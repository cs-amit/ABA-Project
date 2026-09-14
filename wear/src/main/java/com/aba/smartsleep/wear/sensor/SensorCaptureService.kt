package com.aba.smartsleep.wear.sensor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import com.aba.smartsleep.wear.transport.WearBatchSender
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

class SensorCaptureService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var dataSource: SamsungHealthSensorDataSource
    private lateinit var batchSender: WearBatchSender
    private var captureJob: Job? = null
    private var transferJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        dataSource = SamsungHealthSensorDataSource(this, onCaptureFailure = ::stopSelf)
        batchSender = WearBatchSender(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = when (intent?.action) {
        ACTION_START -> startCapture(intent.getStringExtra(EXTRA_SESSION_ID), startId)
        ACTION_STOP -> stopCapture(startId)
        else -> START_NOT_STICKY
    }

    override fun onDestroy() {
        captureJob?.cancel()
        transferJob?.cancel()
        runBlocking(Dispatchers.IO) { batchSender.flush() }
        batchSender.close()
        dataSource.close()
        serviceScope.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startCapture(sessionId: String?, startId: Int): Int {
        if (sessionId.isNullOrBlank()) {
            stopSelf(startId)
            return START_NOT_STICKY
        }

        startForeground(NOTIFICATION_ID, captureNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH)
        captureJob?.cancel()
        transferJob?.cancel()
        captureJob = serviceScope.launch {
            runCatching { dataSource.start(sessionId) }
                .onFailure { stopSelf(startId) }
        }
        transferJob = serviceScope.launch(Dispatchers.IO) {
            batchSender.collect(sessionId, dataSource.samples())
        }
        return START_NOT_STICKY
    }

    private fun stopCapture(startId: Int): Int {
        captureJob?.cancel()
        serviceScope.launch {
            dataSource.stop()
            transferJob?.cancelAndJoin()
            batchSender.flush()
            stopSelf(startId)
        }
        return START_NOT_STICKY
    }

    private fun captureNotification(): Notification {
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Sensor capture",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Smart Sleep is capturing sensors")
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val ACTION_START = "com.aba.smartsleep.wear.action.START_CAPTURE"
        const val ACTION_STOP = "com.aba.smartsleep.wear.action.STOP_CAPTURE"
        const val EXTRA_SESSION_ID = "session_id"

        private const val CHANNEL_ID = "sensor_capture"
        private const val NOTIFICATION_ID = 4

        fun startIntent(context: Context, sessionId: String): Intent =
            Intent(context, SensorCaptureService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_SESSION_ID, sessionId)

        fun stopIntent(context: Context): Intent =
            Intent(context, SensorCaptureService::class.java).setAction(ACTION_STOP)
    }
}
