package com.aba.smartsleep.app.alarm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AlarmCoordinatorTest {
    @Test
    fun `schedules fallback at configured target`() {
        val gateway = FakeAlarmGateway()

        AlarmScheduler(gateway, FixedClock(100)).scheduleFallback("s1", 10_000)

        assertEquals(10_000, gateway.scheduled.single().triggerAtMillis)
    }

    @Test
    fun `rejects fallback target in the past`() {
        val scheduler = AlarmScheduler(FakeAlarmGateway(), FixedClock(10_000))

        try {
            scheduler.scheduleFallback("s1", 9_999)
        } catch (error: IllegalArgumentException) {
            return
        }

        throw AssertionError("Expected a past fallback target to be rejected")
    }

    @Test
    fun `light sleep starts haptics before delayed phone audio`() {
        val output = FakeAlarmOutput()
        val delayScheduler = FakeDelayScheduler()
        val coordinator = AlarmCoordinator(
            sessionId = "s1",
            output = output,
            phoneAudioDelayMillis = 500,
            delayScheduler = delayScheduler,
            claimStore = AlarmClaimStore(),
        )

        assertTrue(coordinator.trigger(AlarmReason.LIGHT_SLEEP))
        assertEquals(1, output.hapticStarts)
        assertEquals(0, output.audioStarts)
        assertEquals(500L, delayScheduler.delayMillis)

        delayScheduler.runScheduledAction()

        assertEquals(1, output.audioStarts)
    }

    @Test
    fun `session claim prevents fallback output after light sleep trigger`() {
        val claimStore = AlarmClaimStore()
        val lightSleepOutput = FakeAlarmOutput()
        val fallbackOutput = FakeAlarmOutput()
        val lightSleep = AlarmCoordinator("s1", lightSleepOutput, claimStore = claimStore)
        val fallback = AlarmCoordinator("s1", fallbackOutput, claimStore = claimStore)

        assertTrue(lightSleep.trigger(AlarmReason.LIGHT_SLEEP))
        assertFalse(fallback.trigger(AlarmReason.FALLBACK))
        assertEquals(1, lightSleepOutput.hapticStarts)
        assertEquals(1, lightSleepOutput.audioStarts)
        assertEquals(0, fallbackOutput.hapticStarts)
        assertEquals(0, fallbackOutput.audioStarts)
    }

    private class FixedClock(private val nowMillis: Long) : Clock {
        override fun currentTimeMillis(): Long = nowMillis
    }

    private class FakeAlarmGateway : AlarmGateway {
        val scheduled = mutableListOf<AlarmRequest>()

        override fun schedule(request: AlarmRequest) {
            scheduled += request
        }
    }

    private class FakeAlarmOutput : AlarmOutput {
        var hapticStarts = 0
        var audioStarts = 0

        override fun startWatchHaptics() {
            hapticStarts += 1
        }

        override fun startPhoneAudio() {
            audioStarts += 1
        }
    }

    private class FakeDelayScheduler : AlarmDelayScheduler {
        var delayMillis: Long? = null
        private var action: (() -> Unit)? = null

        override fun schedule(delayMillis: Long, action: () -> Unit) {
            this.delayMillis = delayMillis
            this.action = action
        }

        fun runScheduledAction() {
            checkNotNull(action).invoke()
        }
    }
}
