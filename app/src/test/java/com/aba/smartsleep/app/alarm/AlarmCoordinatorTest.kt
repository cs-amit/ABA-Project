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
    fun `first trigger wins over all later triggers`() {
        val output = FakeAlarmOutput()
        val coordinator = AlarmCoordinator(output)

        assertTrue(coordinator.trigger(AlarmReason.LIGHT_SLEEP))
        assertFalse(coordinator.trigger(AlarmReason.FALLBACK))
        assertEquals(1, output.hapticStarts)
        assertEquals(1, output.audioStarts)
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
}
