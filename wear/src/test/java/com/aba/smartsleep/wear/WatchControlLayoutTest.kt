package com.aba.smartsleep.wear

import android.view.Gravity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WatchControlLayoutTest {
    @Test
    fun `uses compact bottom centered controls inside circular safe area`() {
        val layout = CircularWatchControlLayout.default()

        assertTrue(layout.buttonHeightDp in 40..44)
        assertEquals(42, layout.buttonHeightDp)
        assertEquals(8, layout.verticalGapDp)
        assertEquals(160, layout.controlWidthDp)
        assertEquals(20, layout.sideInsetDp)
        assertEquals(24, layout.bottomInsetDp)
        assertEquals(Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL, layout.gravity)
    }
}
