package com.aba.smartsleep.wear.haptics

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator

class WatchAlarmController(context: Context) {
    private val vibrator = context.getSystemService(Vibrator::class.java)

    fun vibrate() {
        vibrator?.vibrate(VibrationEffect.createWaveform(VIBRATION_PATTERN, -1))
    }

    fun dismiss() {
        vibrator?.cancel()
    }

    private companion object {
        val VIBRATION_PATTERN = longArrayOf(0, 250, 150, 250, 150, 250)
    }
}
