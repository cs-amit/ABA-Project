package com.aba.smartsleep.app

import android.Manifest
import android.app.Activity
import android.app.NotificationManager
import android.app.TimePickerDialog
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.*
import com.aba.smartsleep.app.alarm.AlarmScheduler
import com.aba.smartsleep.app.alarm.AndroidAlarmGateway
import com.aba.smartsleep.core.model.AlarmSettings
import java.text.DateFormat

class MainActivity : Activity() {
    private lateinit var content: LinearLayout
    private var selectedWindow = 30
    private var selectedHour = 7
    private var selectedMinute = 0
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); showDashboard() }
    private fun showDashboard() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 28, 32, 20) }
        root.addView(TextView(this).apply { text = "Smart Sleep\nClassroom monitoring MVP"; textSize = 25f })
        content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(content, LinearLayout.LayoutParams(-1, 0, 1f))
        val nav = LinearLayout(this).apply { gravity = Gravity.CENTER }
        DashboardTab.entries.forEach { tab -> nav.addView(Button(this).apply { text = tab.label; setOnClickListener { render(tab) } }, LinearLayout.LayoutParams(0, -2, 1f)) }
        root.addView(nav); setContentView(root); render(DashboardTab.LIVE)
    }
    private fun render(tab: DashboardTab) { content.removeAllViews(); when (tab) { DashboardTab.LIVE -> renderLive(); DashboardTab.ALARM -> renderAlarm(); DashboardTab.HISTORY -> renderHistory() } }
    private fun renderLive() {
        val s = (application as SmartSleepApplication).dashboardState.value
        addHeading("Live watch data"); addCard("Watch", if (s.watchLinked) "Connected" else "Waiting for watch data")
        addCard("Samples received", s.samplesReceived.toString()); addCard("Batches / epochs", "${s.batchesReceived} / ${s.epochsGenerated}")
        addCard("Latest sleep probability", s.latestProbability?.let { "${(it * 100).toInt()}%" } ?: "Waiting for 10 valid epochs"); addCard("Session", s.sessionId ?: "No active session")
        content.addView(TextView(this).apply { text = notificationText(); textSize = 13f })
        content.addView(Button(this).apply { text = "Enable alarm notifications"; setOnClickListener { requestNotificationPermission() } })
    }
    private fun renderAlarm() {
        addHeading("Alarm settings")
        val timeButton = Button(this).apply { text = formatTime(); setOnClickListener { TimePickerDialog(this@MainActivity, { _, h, m -> selectedHour = h; selectedMinute = m; text = formatTime() }, selectedHour, selectedMinute, true).show() } }
        content.addView(timeButton); content.addView(TextView(this).apply { text = "Wake window" })
        content.addView(Spinner(this).apply { adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, SUPPORTED_WAKE_WINDOWS.map { "$it minutes" }); setSelection(1); onItemSelectedListener = object : AdapterView.OnItemSelectedListener { override fun onNothingSelected(p: AdapterView<*>?) = Unit; override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { selectedWindow = SUPPORTED_WAKE_WINDOWS[pos] } } })
        content.addView(Button(this).apply { text = "Save alarm"; setOnClickListener { val trigger = nextAlarmEpochMillis(System.currentTimeMillis(), selectedHour, selectedMinute); AlarmScheduler(AndroidAlarmGateway(this@MainActivity)).scheduleFallback("classroom-$trigger", AlarmSettings(trigger, selectedWindow)); Toast.makeText(this@MainActivity, "Fallback alarm scheduled for ${DateFormat.getTimeInstance(DateFormat.SHORT).format(trigger)}", Toast.LENGTH_LONG).show() } })
        addCard("Safety", "Phone fallback is always scheduled at the selected time")
    }
    private fun renderHistory() { val s = (application as SmartSleepApplication).dashboardState.value; addHeading("History"); addCard("Current session", s.sessionId ?: "No session yet"); addCard("Data captured", "${s.samplesReceived} samples in ${s.batchesReceived} batches"); addCard("Prediction", s.latestProbability?.let { "${(it * 100).toInt()}% latest probability" } ?: "No prediction recorded yet"); addCard("Storage", "Kept locally for this classroom MVP") }
    private fun addHeading(t: String) = content.addView(TextView(this).apply { text = t; textSize = 21f; setPadding(0, 18, 0, 12) })
    private fun addCard(l: String, v: String) = content.addView(TextView(this).apply { text = "$l\n$v"; textSize = 16f; setPadding(16, 14, 16, 14) })
    private fun formatTime() = String.format("Alarm: %02d:%02d", selectedHour, selectedMinute)
    private fun notificationText() = if (getSystemService(NotificationManager::class.java).areNotificationsEnabled()) "Alarm notifications enabled" else "Alarm notifications disabled"
    private fun requestNotificationPermission() { if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1) else render(DashboardTab.LIVE) }
}
