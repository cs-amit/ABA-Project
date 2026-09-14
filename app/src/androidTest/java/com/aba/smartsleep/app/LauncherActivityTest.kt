package com.aba.smartsleep.app

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LauncherActivityTest {
    @Test
    fun appPackageResolvesLauncherActivity() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val launcherIntent = Intent(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER)
            .setPackage(context.packageName)

        val resolvedActivity = context.packageManager.resolveActivity(launcherIntent, 0)

        assertNotNull("The app package must resolve a launcher activity", resolvedActivity)
        assertEquals(context.packageName, resolvedActivity?.activityInfo?.packageName)
    }
}
