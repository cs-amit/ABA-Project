package com.aba.smartsleep.app

import android.Manifest
import android.app.Activity
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var notificationStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        notificationStatus = TextView(this)
        val enableNotificationsButton = Button(this).apply {
            text = "Enable alarm notifications"
            contentDescription = "Enable alarm notifications"
            setOnClickListener(::requestNotificationPermission)
        }
        setContentView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                addView(TextView(context).apply { text = "Smart Sleep setup" })
                addView(notificationStatus)
                addView(enableNotificationsButton)
            },
        )
    }

    override fun onResume() {
        super.onResume()
        refreshNotificationStatus()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == NOTIFICATION_PERMISSION_REQUEST_CODE) {
            refreshNotificationStatus()
        }
    }

    private fun requestNotificationPermission(view: android.view.View) {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                NOTIFICATION_PERMISSION_REQUEST_CODE,
            )
        } else {
            refreshNotificationStatus()
        }
    }

    private fun refreshNotificationStatus() {
        val notificationsEnabled = getSystemService(NotificationManager::class.java).areNotificationsEnabled()
        notificationStatus.text = if (notificationsEnabled) {
            "Alarm notifications are enabled"
        } else {
            "Alarm notifications are disabled"
        }
    }

    private companion object {
        const val NOTIFICATION_PERMISSION_REQUEST_CODE = 1
    }
}
