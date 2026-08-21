package com.aba.smartsleep.app.alarm

import java.util.concurrent.atomic.AtomicBoolean

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

class AlarmCoordinator(
    private val output: AlarmOutput,
    private val phoneAudioDelayMillis: Long = 0,
    private val delayScheduler: AlarmDelayScheduler = AlarmDelayScheduler { _, action -> action() },
) {
    private val triggered = AtomicBoolean(false)

    fun trigger(reason: AlarmReason): Boolean {
        if (!triggered.compareAndSet(false, true)) {
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
