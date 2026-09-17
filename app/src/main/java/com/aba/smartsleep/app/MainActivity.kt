package com.aba.smartsleep.app

import android.Manifest
import android.app.Activity
import android.app.TimePickerDialog
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.*
import com.aba.smartsleep.app.alarm.AlarmScheduler
import com.aba.smartsleep.app.alarm.AndroidAlarmGateway
import com.aba.smartsleep.app.inference.DEMO_BUTTON_LABEL
import com.aba.smartsleep.core.model.AlarmSettings
import java.text.DateFormat
import java.util.Locale
import kotlinx.coroutines.*

class MainActivity : Activity() {
    private lateinit var content: LinearLayout
    private lateinit var alarmPreferences: SharedPreferences
    private var liveViews: LiveViews? = null
    private val navButtons = mutableMapOf<DashboardTab, Button>()
    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val alarmScheduler by lazy { AlarmScheduler(AndroidAlarmGateway(this)) }
    private var selectedWindow = 30
    private var selectedHour = 7
    private var selectedMinute = 0
    private var currentTab = DashboardTab.LIVE

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        alarmPreferences = getSharedPreferences(ALARM_PREFERENCES, MODE_PRIVATE)
        selectedHour = alarmPreferences.getInt(KEY_HOUR, selectedHour)
        selectedMinute = alarmPreferences.getInt(KEY_MINUTE, selectedMinute)
        selectedWindow = alarmPreferences.getInt(KEY_WINDOW, selectedWindow)
        window.statusBarColor = NAVY
        showDashboard()
        uiScope.launch {
            smartSleepApplication.dashboardState.collect { state ->
                if (currentTab == DashboardTab.LIVE) updateLive(state)
            }
        }
    }

    override fun onDestroy() {
        uiScope.cancel()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == NOTIFICATION_PERMISSION_REQUEST && currentTab == DashboardTab.LIVE) render(DashboardTab.LIVE)
    }

    private fun showDashboard() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(PAGE_BACKGROUND)
        }
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(20), dp(22), dp(18))
            setBackgroundColor(NAVY)
            addView(TextView(this@MainActivity).apply {
                text = "Smart Sleep"
                textSize = 27f
                setTextColor(Color.WHITE)
                setTypeface(typeface, Typeface.BOLD)
            })
            addView(TextView(this@MainActivity).apply {
                text = "Classroom sleep monitoring"
                textSize = 14f
                setTextColor(SUBTITLE)
            })
        })
        content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(20))
        }
        root.addView(ScrollView(this).apply {
            isFillViewport = true
            addView(content)
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(LinearLayout(this).apply {
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(5), dp(8), dp(7))
            setBackgroundColor(Color.WHITE)
            DashboardTab.entries.forEach { tab ->
                addView(Button(this@MainActivity).apply {
                    text = tab.label
                    textSize = 13f
                    setTextColor(NAVY)
                    backgroundTintList = ColorStateList.valueOf(Color.WHITE)
                    setOnClickListener { render(tab) }
                    navButtons[tab] = this
                }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            }
        })
        setContentView(root)
        render(DashboardTab.LIVE)
    }

    private fun render(tab: DashboardTab) {
        currentTab = tab
        liveViews = null
        content.removeAllViews()
        navButtons.forEach { (buttonTab, button) ->
            button.backgroundTintList = ColorStateList.valueOf(if (buttonTab == tab) PALE_BLUE else Color.WHITE)
            button.setTypeface(button.typeface, if (buttonTab == tab) Typeface.BOLD else Typeface.NORMAL)
        }
        when (tab) {
            DashboardTab.LIVE -> renderLive()
            DashboardTab.ALARM -> renderAlarm()
            DashboardTab.HISTORY -> renderHistory()
        }
    }

    private fun renderLive() {
        addHeading("Live data", "Updates once for each completed 30-second epoch")
        liveViews = LiveViews(addCard(), addCard(), addCard(), addCard())
        updateLive(smartSleepApplication.dashboardState.value)
        content.addView(Button(this).apply {
            text = DEMO_BUTTON_LABEL
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(ACCENT)
            setOnClickListener {
                isEnabled = false
                smartSleepApplication.runDemoInference()
                postDelayed({ isEnabled = true }, 750L)
                Toast.makeText(this@MainActivity, "Synthetic demo only — no alarm is scheduled", Toast.LENGTH_LONG).show()
            }
        }, standardMargins())
        if (notificationPermissionNeeded(Build.VERSION.SDK_INT, notificationPermissionGranted())) {
            content.addView(Button(this).apply {
                text = "Allow alarm notifications"
                setTextColor(Color.WHITE)
                backgroundTintList = ColorStateList.valueOf(ACCENT)
                setOnClickListener { requestNotificationPermission() }
            }, standardMargins())
        }
    }

    private fun updateLive(state: DashboardState) {
        val views = liveViews ?: return
        views.watch.text = if (state.watchLinked) {
            "WATCH CONNECTION\nConnected • ${state.batchesReceived} batches • ${state.samplesReceived} samples"
        } else "WATCH CONNECTION\nWaiting for watch data"
        val demo = state.demoResult
        if (demo != null) {
            views.progress.text = "DEMO WINDOW • SYNTHETIC DATA\n${demo.epochsProcessed} / 10 epochs\nPassed through the same ONNX pipeline"
            views.epoch.text = "DEMO DATA — NOT FROM WATCH\n" + formatEpoch(demo.latestEpoch.toEpochSummary())
            views.prediction.text = if (demo.inferenceAvailable) {
                val thresholdStatus = if (demo.aboveAlarmThreshold) "Above alarm threshold" else "Below alarm threshold"
                "DEMO ONNX RESULT\n${formatSleepProbability(demo.probability)} sleep probability\n$thresholdStatus • no real alarm scheduled"
            } else {
                "DEMO ONNX RESULT\nModel unavailable • no real alarm scheduled"
            }
            return
        }
        views.progress.text = "PREDICTION WINDOW\n${state.epochsInCurrentWindow} / 10 epochs\nA prediction is made after ten valid epochs"
        views.epoch.text = state.latestEpoch?.let(::formatEpoch) ?: "LATEST EPOCH\nWaiting for the first complete epoch"
        views.prediction.text = state.latestProbability?.let {
            val prefix = if (state.epochsInCurrentWindow == 10) "NEW ONNX PREDICTION" else "LATEST ONNX PREDICTION"
            "$prefix\n${formatSleepProbability(it)} sleep probability"
        } ?: "ONNX PREDICTION\nWaiting for ten valid epochs"
    }

    private fun formatEpoch(epoch: EpochSummary): String {
        val time = DateFormat.getTimeInstance(DateFormat.MEDIUM).format(epoch.endEpochMillis)
        val quality = if (epoch.validForInference) "Ready for prediction" else "Calibration or low sensor coverage"
        return String.format(
            Locale.US,
            "LATEST EPOCH • %s\nHeart rate %.1f bpm • variability %.1f bpm\nMovement %.0f changes\nCoverage: motion %d%% • heart %d%%\n%s",
            time, epoch.heartRateMean, epoch.heartRateVariability, epoch.activityCount,
            epoch.motionCoveragePercent, epoch.heartRateCoveragePercent, quality,
        )
    }

    private fun renderAlarm() {
        addHeading("Alarm", "Choose a target time and smart wake window")
        val status = addCard()
        fun refreshStatus() {
            status.text = if (alarmPreferences.getBoolean(KEY_ENABLED, false)) {
                "ALARM ACTIVE\n${formatTime()} • $selectedWindow minute wake window"
            } else "ALARM OFF\nNo fallback alarm is scheduled"
        }
        refreshStatus()
        content.addView(Switch(this).apply {
            text = "Alarm enabled"
            textSize = 17f
            setTextColor(NAVY)
            isChecked = alarmPreferences.getBoolean(KEY_ENABLED, false)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            setOnCheckedChangeListener { _, enabled ->
                alarmPreferences.edit().putBoolean(KEY_ENABLED, enabled).apply()
                if (enabled) {
                    scheduleSelectedAlarm(showConfirmation = true)
                } else {
                    alarmScheduler.cancelFallback(ALARM_SESSION_ID)
                    Toast.makeText(this@MainActivity, "Alarm cancelled", Toast.LENGTH_SHORT).show()
                }
                refreshStatus()
            }
        }, standardMargins())
        content.addView(Button(this).apply {
            text = formatTime()
            backgroundTintList = ColorStateList.valueOf(PALE_BLUE)
            setTextColor(NAVY)
            setOnClickListener {
                TimePickerDialog(this@MainActivity, { _, hour, minute ->
                    selectedHour = hour
                    selectedMinute = minute
                    text = formatTime()
                    saveAlarmChoices()
                    refreshStatus()
                }, selectedHour, selectedMinute, true).show()
            }
        }, standardMargins())
        content.addView(TextView(this).apply {
            text = "Wake window"
            textSize = 14f
            setTextColor(MUTED)
            setPadding(dp(12), dp(10), 0, 0)
        })
        content.addView(Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, SUPPORTED_WAKE_WINDOWS.map { "$it minutes" })
            setSelection(SUPPORTED_WAKE_WINDOWS.indexOf(selectedWindow).coerceAtLeast(0))
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    selectedWindow = SUPPORTED_WAKE_WINDOWS[position]
                    saveAlarmChoices()
                    refreshStatus()
                }
            }
        }, standardMargins())
        content.addView(Button(this).apply {
            text = "Save alarm settings"
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(ACCENT)
            setOnClickListener {
                saveAlarmChoices()
                if (alarmPreferences.getBoolean(KEY_ENABLED, false)) scheduleSelectedAlarm(showConfirmation = true)
                else Toast.makeText(this@MainActivity, "Settings saved. Turn on the alarm when ready.", Toast.LENGTH_LONG).show()
                refreshStatus()
            }
        }, standardMargins())
        addInfo("Phone fallback", "The phone alarm remains the safety backup at the selected target time.")
    }

    private fun renderHistory() {
        val state = smartSleepApplication.dashboardState.value
        addHeading("Session summary", "Locally stored classroom-MVP activity")
        addInfo("Current session", state.sessionId ?: "No session yet")
        addInfo("Data captured", "${state.samplesReceived} samples • ${state.batchesReceived} batches • ${state.epochsGenerated} epochs")
        addInfo("Latest prediction", state.latestProbability?.let { "${formatSleepProbability(it)} sleep probability" } ?: "No prediction recorded yet")
        addInfo("Storage", "Sensor features and predictions stay on this phone.")
    }

    private fun addHeading(title: String, subtitle: String) {
        content.addView(TextView(this).apply {
            text = title
            textSize = 23f
            setTextColor(NAVY)
            setTypeface(typeface, Typeface.BOLD)
            setPadding(dp(4), dp(14), dp(4), 0)
        })
        content.addView(TextView(this).apply {
            text = subtitle
            textSize = 14f
            setTextColor(MUTED)
            setPadding(dp(4), dp(2), dp(4), dp(8))
        })
    }

    private fun addCard(): TextView = TextView(this).apply {
        textSize = 15f
        setTextColor(NAVY)
        setLineSpacing(0f, 1.15f)
        setPadding(dp(16), dp(15), dp(16), dp(15))
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(14).toFloat()
            setColor(Color.WHITE)
            setStroke(dp(1), CARD_BORDER)
        }
        content.addView(this, standardMargins())
    }

    private fun addInfo(label: String, value: String) {
        addCard().text = "${label.uppercase(Locale.US)}\n$value"
    }

    private fun standardMargins() = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
        setMargins(0, dp(5), 0, dp(5))
    }

    private fun scheduleSelectedAlarm(showConfirmation: Boolean) {
        val trigger = nextAlarmEpochMillis(System.currentTimeMillis(), selectedHour, selectedMinute)
        alarmScheduler.scheduleFallback(ALARM_SESSION_ID, AlarmSettings(trigger, selectedWindow))
        alarmPreferences.edit().putLong(KEY_TRIGGER, trigger).apply()
        if (showConfirmation) {
            val time = DateFormat.getTimeInstance(DateFormat.SHORT).format(trigger)
            Toast.makeText(this, "Alarm scheduled for $time", Toast.LENGTH_LONG).show()
        }
    }

    private fun saveAlarmChoices() {
        alarmPreferences.edit().putInt(KEY_HOUR, selectedHour).putInt(KEY_MINUTE, selectedMinute).putInt(KEY_WINDOW, selectedWindow).apply()
    }

    private fun formatTime(): String = String.format(Locale.US, "Target time  %02d:%02d", selectedHour, selectedMinute)
    private fun notificationPermissionGranted(): Boolean = Build.VERSION.SDK_INT < 33 ||
        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
    private fun requestNotificationPermission() {
        if (notificationPermissionNeeded(Build.VERSION.SDK_INT, notificationPermissionGranted())) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), NOTIFICATION_PERMISSION_REQUEST)
        }
    }
    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
    private val smartSleepApplication: SmartSleepApplication get() = application as SmartSleepApplication
    private data class LiveViews(val watch: TextView, val progress: TextView, val epoch: TextView, val prediction: TextView)

    private companion object {
        const val NOTIFICATION_PERMISSION_REQUEST = 1
        const val ALARM_PREFERENCES = "alarm-ui"
        const val ALARM_SESSION_ID = "classroom-alarm"
        const val KEY_ENABLED = "enabled"
        const val KEY_HOUR = "hour"
        const val KEY_MINUTE = "minute"
        const val KEY_WINDOW = "window"
        const val KEY_TRIGGER = "trigger"
        val NAVY = Color.rgb(19, 36, 61)
        val ACCENT = Color.rgb(45, 109, 246)
        val PALE_BLUE = Color.rgb(226, 236, 255)
        val PAGE_BACKGROUND = Color.rgb(246, 248, 252)
        val CARD_BORDER = Color.rgb(223, 228, 237)
        val MUTED = Color.rgb(91, 104, 124)
        val SUBTITLE = Color.rgb(194, 211, 237)
    }
}
