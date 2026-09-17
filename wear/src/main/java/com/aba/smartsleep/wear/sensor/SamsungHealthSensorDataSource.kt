package com.aba.smartsleep.wear.sensor

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Vibrator
import android.util.Log
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.model.WearableCapability
import com.aba.smartsleep.core.model.WearableDataSource
import com.samsung.android.service.health.tracking.ConnectionListener
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.ValueKey
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow

class SamsungHealthSensorDataSource(
    context: Context,
    private val onCaptureFailure: () -> Unit = {},
) : WearableDataSource {
    private val appContext = context.applicationContext
    private val stateLock = Any()
    private val sampleHandoff = RetainedSampleHandoff(SAMPLE_BUFFER_CAPACITY)
    private val activeTrackers = mutableMapOf<HealthTrackerType, HealthTracker>()
    private val flushScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val trackerFlusher = ActiveTrackerFlusher(flushScope, ::flushHeartRate)

    private val capabilityState = CaptureCapabilityState(hapticCapability())
    private var activeSessionId: String? = null
    private var connectionRequested = false
    private var serviceConnected = false

    private val healthTrackingService = HealthTrackingService(
        object : ConnectionListener {
            override fun onConnectionSuccess() {
                onServiceConnected()
            }

            override fun onConnectionEnded() {
                handleCaptureFailure()
            }

            override fun onConnectionFailed(exception: HealthTrackerException) {
                handleCaptureFailure()
            }
        },
        appContext,
    )

    override suspend fun capabilities(): Set<WearableCapability> = synchronized(stateLock) {
        capabilityState.capabilities()
    }

    override fun samples(): Flow<SensorSample> = sampleHandoff.samples()

    override suspend fun start(sessionId: String) {
        require(sessionId.isNotBlank()) { "A non-blank session ID is required for sensor capture." }
        requireCapturePermissions()

        val shouldConnect = synchronized(stateLock) {
            activeSessionId = sessionId
            if (serviceConnected || connectionRequested) {
                false
            } else {
                connectionRequested = true
                true
            }
        }

        if (serviceConnected) {
            startSupportedTrackers()
        } else if (shouldConnect) {
            healthTrackingService.connectService()
        }
    }

    override suspend fun stop() {
        synchronized(stateLock) {
            activeSessionId = null
            stopTrackersLocked()
        }
    }

    fun close() {
        synchronized(stateLock) {
            activeSessionId = null
            stopTrackersLocked()
            serviceConnected = false
            connectionRequested = false
            capabilityState.onCaptureFailure()
        }
        healthTrackingService.disconnectService()
        flushScope.cancel()
    }

    private fun onServiceConnected() {
        val trackerTypes = healthTrackingService.trackingCapability.supportHealthTrackerTypes.toSet()
        Log.i(TAG, "Supported Samsung trackers: $trackerTypes")
        synchronized(stateLock) {
            serviceConnected = true
            capabilityState.onServiceConnected(trackerTypes)
        }
        startSupportedTrackers()
    }

    private fun startSupportedTrackers() {
        try {
            synchronized(stateLock) {
                if (activeSessionId == null) return
                startTrackerLocked(
                    HealthTrackerType.ACCELEROMETER_CONTINUOUS,
                    accelerometerListener,
                )
                startTrackerLocked(
                    HealthTrackerType.HEART_RATE_CONTINUOUS,
                    heartRateListener,
                )
                trackerFlusher.start()
            }
        } catch (error: RuntimeException) {
            handleCaptureFailure()
        }
    }

    private fun startTrackerLocked(
        trackerType: HealthTrackerType,
        listener: HealthTracker.TrackerEventListener,
    ) {
        if (!capabilityState.supports(trackerType) || trackerType in activeTrackers) return

        val tracker = requireNotNull(healthTrackingService.getHealthTracker(trackerType)) {
            "Samsung Health Sensor Service did not provide $trackerType."
        }
        tracker.setEventListener(listener)
        activeTrackers[trackerType] = tracker
    }

    private fun stopTrackersLocked() {
        trackerFlusher.stop()
        activeTrackers.values.forEach(HealthTracker::unsetEventListener)
        activeTrackers.clear()
    }

    private fun flushHeartRate() {
        synchronized(stateLock) {
            if (activeSessionId == null) return
            val tracker = activeTrackers[HealthTrackerType.HEART_RATE_CONTINUOUS] ?: return
            try {
                if (!tracker.flush()) Log.w(TAG, "Heart-rate flush was not accepted; retrying next interval.")
            } catch (error: RuntimeException) {
                Log.w(TAG, "Heart-rate flush failed; retrying next interval.", error)
            }
        }
    }

    private val accelerometerListener = object : HealthTracker.TrackerEventListener {
        override fun onDataReceived(dataPoints: List<DataPoint>) {
            dataPoints.forEach { dataPoint ->
                val x = dataPoint.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_X) ?: return@forEach
                val y = dataPoint.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Y) ?: return@forEach
                val z = dataPoint.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Z) ?: return@forEach
                sampleHandoff.emit(
                    SamsungSampleMapper.accelerometer(
                        timestamp = dataPoint.timestamp,
                        x = x,
                        y = y,
                        z = z,
                    ),
                )
            }
        }

        override fun onFlushCompleted() = Unit

        override fun onError(error: HealthTracker.TrackerError) {
            handleCaptureFailure()
        }
    }

    private val heartRateListener = object : HealthTracker.TrackerEventListener {
        override fun onDataReceived(dataPoints: List<DataPoint>) {
            dataPoints.forEach { dataPoint ->
                val bpm = dataPoint.getValue(ValueKey.HeartRateSet.HEART_RATE) ?: return@forEach
                val status = dataPoint.getValue(ValueKey.HeartRateSet.HEART_RATE_STATUS)
                    ?: UNKNOWN_HEART_RATE_STATUS
                val ibiValues = dataPoint.getValue(ValueKey.HeartRateSet.IBI_LIST).orEmpty()

                if (ibiValues.isEmpty()) {
                    emitHeartRate(dataPoint.timestamp, bpm, null, status)
                } else {
                    ibiValues.forEach { ibi ->
                        emitHeartRate(dataPoint.timestamp, bpm, ibi, status)
                    }
                }
            }
        }

        override fun onFlushCompleted() = Unit

        override fun onError(error: HealthTracker.TrackerError) {
            handleCaptureFailure()
        }
    }

    private fun emitHeartRate(timestamp: Long, bpm: Int, ibi: Int?, status: Int) {
        sampleHandoff.emit(
            SamsungSampleMapper.heartRate(
                timestamp = timestamp,
                bpm = bpm.toFloat(),
                ibi = ibi,
                status = status,
            ),
        )
    }

    private fun handleCaptureFailure() {
        Log.w(TAG, "Samsung sensor capture stopped after a service or tracker failure.")
        sampleHandoff.emit(
            SensorSample(
                timestampEpochMillis = System.currentTimeMillis(),
                accelX = null,
                accelY = null,
                accelZ = null,
                heartRateBpm = null,
                ibiMillis = null,
                quality = SensorQuality.UNAVAILABLE,
            ),
        )
        synchronized(stateLock) {
            activeSessionId = null
            stopTrackersLocked()
            serviceConnected = false
            connectionRequested = false
            capabilityState.onCaptureFailure()
        }
        onCaptureFailure()
    }

    private fun requireCapturePermissions() {
        val missingPermissions = REQUIRED_PERMISSIONS.filter { permission ->
            appContext.checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED
        }
        check(missingPermissions.isEmpty()) {
            "Request ACTIVITY_RECOGNITION and BODY_SENSORS before sensor capture."
        }
    }

    private fun hapticCapability(): Set<WearableCapability> =
        if (appContext.getSystemService(Vibrator::class.java)?.hasVibrator() == true) {
            setOf(WearableCapability.HAPTICS)
        } else {
            emptySet()
        }

    private companion object {
        const val TAG = "SamsungHealthSensor"
        const val SAMPLE_BUFFER_CAPACITY = 2_048
        const val UNKNOWN_HEART_RATE_STATUS = Int.MIN_VALUE
        val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.ACTIVITY_RECOGNITION,
            Manifest.permission.BODY_SENSORS,
        )
    }
}

internal class RetainedSampleHandoff(capacity: Int) {
    private val samples = Channel<SensorSample>(
        capacity = capacity,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    fun emit(sample: SensorSample) {
        samples.trySend(sample)
    }

    fun samples(): Flow<SensorSample> = samples.receiveAsFlow()
}

internal class CaptureCapabilityState(
    private val hapticCapabilities: Set<WearableCapability>,
) {
    private var supportedTrackerTypes: Set<HealthTrackerType> = emptySet()
    private var supportedCapabilities: Set<WearableCapability> = hapticCapabilities

    fun capabilities(): Set<WearableCapability> = supportedCapabilities

    fun supports(trackerType: HealthTrackerType): Boolean = trackerType in supportedTrackerTypes

    fun onServiceConnected(trackerTypes: Set<HealthTrackerType>) {
        supportedTrackerTypes = trackerTypes
        supportedCapabilities = buildSet {
            addAll(hapticCapabilities)
            if (HealthTrackerType.ACCELEROMETER_CONTINUOUS in trackerTypes) {
                add(WearableCapability.ACCELEROMETER)
            }
            if (HealthTrackerType.HEART_RATE_CONTINUOUS in trackerTypes) {
                add(WearableCapability.HEART_RATE_WITH_IBI)
            }
        }
    }

    fun onCaptureFailure() {
        supportedTrackerTypes = emptySet()
        supportedCapabilities = hapticCapabilities
    }
}

internal object SamsungSampleMapper {
    private const val OFF_BODY_HEART_RATE_STATUS = -3
    // Samsung Health Sensor SDK: 1 means a successful measurement; 0 is initial measuring.
    private const val VALID_HEART_RATE_STATUS = 1

    fun heartRate(
        timestamp: Long,
        bpm: Float,
        ibi: Int?,
        status: Int,
    ): SensorSample = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = null,
        accelY = null,
        accelZ = null,
        heartRateBpm = bpm,
        ibiMillis = ibi,
        quality = when (status) {
            OFF_BODY_HEART_RATE_STATUS -> SensorQuality.OFF_BODY
            VALID_HEART_RATE_STATUS -> SensorQuality.VALID
            else -> SensorQuality.DEGRADED
        },
    )

    fun accelerometer(
        timestamp: Long,
        x: Int,
        y: Int,
        z: Int,
    ): SensorSample = SensorSample(
        timestampEpochMillis = timestamp,
        accelX = x.toFloat(),
        accelY = y.toFloat(),
        accelZ = z.toFloat(),
        heartRateBpm = null,
        ibiMillis = null,
        quality = SensorQuality.VALID,
    )
}
