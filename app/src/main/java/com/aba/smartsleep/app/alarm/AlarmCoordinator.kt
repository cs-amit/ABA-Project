package com.aba.smartsleep.app.alarm

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.ConcurrentHashMap

enum class AlarmReason {
    LIGHT_SLEEP,
    FALLBACK,
}

interface AlarmOutput {
    fun startWatchHaptics()
    fun startPhoneAudio()
}

fun interface AlarmDelayScheduler {
    fun schedule(delayMillis: Long, action: () -> Unit)
}

class AlarmClaimStore {
    private val claims = ConcurrentHashMap<String, AtomicBoolean>()

    fun claim(sessionId: String): Boolean =
        claims.getOrPut(sessionId) { AtomicBoolean(false) }.compareAndSet(false, true)
}

object SharedAlarmClaims {
    val store = AlarmClaimStore()
}

class AlarmCoordinator(
    private val sessionId: String,
    private val output: AlarmOutput,
    private val phoneAudioDelayMillis: Long = 0,
    private val delayScheduler: AlarmDelayScheduler = AlarmDelayScheduler { _, action -> action() },
    private val claimStore: AlarmClaimStore = SharedAlarmClaims.store,
) {
    fun trigger(reason: AlarmReason): Boolean {
        if (!claimStore.claim(sessionId)) {
            return false
        }

        output.startWatchHaptics()
        if (reason == AlarmReason.FALLBACK || phoneAudioDelayMillis == 0L) {
            output.startPhoneAudio()
        } else {
            delayScheduler.schedule(phoneAudioDelayMillis, output::startPhoneAudio)
        }
        return true
    }
}
