package com.aba.smartsleep.wear

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowInsets
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Space
import com.aba.smartsleep.wear.haptics.WatchAlarmController
import com.aba.smartsleep.wear.sensor.SensorCaptureService
import java.util.UUID

class WearMainActivity : Activity() {
    private lateinit var alarmController: WatchAlarmController
    private var startCaptureAfterPermissionGrant = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        alarmController = WatchAlarmController(this)
        setContentView(captureControls())
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAPTURE_PERMISSION_REQUEST && startCaptureAfterPermissionGrant) {
            startCaptureAfterPermissionGrant = false
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                startCapture()
            }
        }
    }

    private fun captureControls(): FrameLayout {
        val layout = CircularWatchControlLayout.default()
        return FrameLayout(this).apply {
            setSafeAreaPadding(layout)
            addView(
                controlColumn(layout),
                FrameLayout.LayoutParams(
                    dp(layout.controlWidthDp),
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    layout.gravity,
                ),
            )
        }
    }

    private fun controlColumn(layout: CircularWatchControlLayout): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        addView(controlButton("Start sensor capture", ::requestPermissionsThenStart, layout))
        addView(verticalGap(layout))
        addView(controlButton("Stop sensor capture", ::stopCapture, layout))
        addView(verticalGap(layout))
        addView(controlButton("Dismiss alarm", alarmController::dismiss, layout))
    }

    private fun controlButton(
        label: String,
        action: () -> Unit,
        layout: CircularWatchControlLayout,
    ): Button = Button(this).apply {
        text = label
        contentDescription = label
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            dp(layout.buttonHeightDp),
        )
        setOnClickListener { action() }
    }

    private fun verticalGap(layout: CircularWatchControlLayout): Space = Space(this).apply {
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            dp(layout.verticalGapDp),
        )
    }

    private fun FrameLayout.setSafeAreaPadding(layout: CircularWatchControlLayout) {
        fun apply(systemInsets: android.graphics.Insets) {
            setPadding(
                dp(layout.sideInsetDp) + systemInsets.left,
                systemInsets.top,
                dp(layout.sideInsetDp) + systemInsets.right,
                dp(layout.bottomInsetDp) + systemInsets.bottom,
            )
        }

        apply(android.graphics.Insets.NONE)
        setOnApplyWindowInsetsListener { _, windowInsets ->
            apply(windowInsets.getInsets(WindowInsets.Type.systemBars()))
            windowInsets
        }
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    private fun requestPermissionsThenStart() {
        if (REQUIRED_PERMISSIONS.all(::hasPermission)) {
            startCapture()
        } else {
            startCaptureAfterPermissionGrant = true
            requestPermissions(REQUIRED_PERMISSIONS, CAPTURE_PERMISSION_REQUEST)
        }
    }

    private fun hasPermission(permission: String): Boolean =
        checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun startCapture() {
        val intent = SensorCaptureService.startIntent(this, UUID.randomUUID().toString())
        startForegroundService(intent)
    }

    private fun stopCapture() {
        startService(SensorCaptureService.stopIntent(this))
    }

    private companion object {
        const val CAPTURE_PERMISSION_REQUEST = 4
        val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.ACTIVITY_RECOGNITION,
            Manifest.permission.BODY_SENSORS,
        )
    }
}

internal data class CircularWatchControlLayout(
    val buttonHeightDp: Int,
    val verticalGapDp: Int,
    val controlWidthDp: Int,
    val sideInsetDp: Int,
    val bottomInsetDp: Int,
    val gravity: Int,
) {
    companion object {
        fun default(): CircularWatchControlLayout = CircularWatchControlLayout(
            buttonHeightDp = 42,
            verticalGapDp = 8,
            controlWidthDp = 160,
            sideInsetDp = 20,
            bottomInsetDp = 24,
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL,
        )
    }
}
