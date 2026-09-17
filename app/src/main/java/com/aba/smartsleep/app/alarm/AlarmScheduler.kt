package com.aba.smartsleep.app.alarm

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import com.aba.smartsleep.core.model.AlarmSettings

data class AlarmRequest(
    val sessionId: String,
    val triggerAtMillis: Long,
)

interface AlarmGateway {
    fun schedule(request: AlarmRequest)
    fun cancel(sessionId: String)
}

fun interface Clock {
    fun currentTimeMillis(): Long
}

class AlarmScheduler(
    private val gateway: AlarmGateway,
    private val clock: Clock = Clock { System.currentTimeMillis() },
) {
    fun scheduleFallback(sessionId: String, targetEpochMillis: Long) {
        require(targetEpochMillis >= clock.currentTimeMillis()) {
            "Fallback alarm target must not be in the past"
        }
        gateway.schedule(AlarmRequest(sessionId, targetEpochMillis))
    }

    fun scheduleFallback(sessionId: String, settings: AlarmSettings) {
        scheduleFallback(sessionId, settings.targetEpochMillis)
    }

    fun cancelFallback(sessionId: String) {
        gateway.cancel(sessionId)
    }
}

class AndroidAlarmGateway(context: Context) : AlarmGateway {
    private val appContext = context.applicationContext
    private val alarmManager = appContext.getSystemService(AlarmManager::class.java)

    override fun schedule(request: AlarmRequest) {
        val operation = fallbackOperation(request.sessionId)
        val showIntent = PendingIntent.getActivity(
            appContext,
            request.sessionId.hashCode() xor DISPLAY_REQUEST_CODE_MASK,
            AlarmActivity.displayIntent(appContext, request.sessionId),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        alarmManager.setAlarmClock(
            AlarmManager.AlarmClockInfo(request.triggerAtMillis, showIntent),
            operation,
        )
    }

    override fun cancel(sessionId: String) {
        fallbackOperation(sessionId).also { operation ->
            alarmManager.cancel(operation)
            operation.cancel()
        }
    }

    private fun fallbackOperation(sessionId: String): PendingIntent = PendingIntent.getBroadcast(
        appContext,
        sessionId.hashCode(),
        AlarmReceiver.fallbackIntent(appContext, sessionId),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )

    private companion object {
        const val DISPLAY_REQUEST_CODE_MASK = 0x51A7
    }
}
