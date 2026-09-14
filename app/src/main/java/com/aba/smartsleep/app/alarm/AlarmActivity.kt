package com.aba.smartsleep.app.alarm

import android.app.Activity
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class AlarmActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sessionId = intent.getStringExtra(EXTRA_SESSION_ID).orEmpty()
        val dismissButton = Button(this).apply {
            text = "Dismiss alarm"
            contentDescription = "Dismiss alarm"
            setOnClickListener {
                AlarmAudio.stop()
                getSystemService(NotificationManager::class.java).cancel(sessionId.hashCode())
                finish()
            }
        }
        setContentView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                addView(TextView(context).apply { text = "Smart Sleep alarm" })
                addView(dismissButton)
            },
        )
    }

    companion object {
        private const val EXTRA_SESSION_ID = "session_id"

        fun displayIntent(context: Context, sessionId: String): Intent =
            Intent(context, AlarmActivity::class.java).putExtra(EXTRA_SESSION_ID, sessionId)
    }
}
