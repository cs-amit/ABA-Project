package com.aba.smartsleep.app

import android.app.Application
import android.util.Log
import androidx.room.Room
import com.aba.smartsleep.app.data.AppDatabase
import com.aba.smartsleep.app.data.RoomSessionRepository
import com.aba.smartsleep.app.data.PredictionEventEntity
import com.aba.smartsleep.app.inference.LiveInferenceProcessor
import com.aba.smartsleep.app.inference.OnnxProbabilityModel
import com.aba.smartsleep.app.transport.PhoneBatchDataLayerRecovery
import com.aba.smartsleep.app.transport.WearableReceiver
import com.aba.smartsleep.app.transport.WatchAcknowledgementRetryCoordinator
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

class SmartSleepApplication : Application() {
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    lateinit var sessionRepository: RoomSessionRepository
        private set
    private lateinit var acknowledgementRetry: WatchAcknowledgementRetryCoordinator

    override fun onCreate() {
        super.onCreate()
        WearableReceiver.initialize(filesDir)
        val liveInference = LiveInferenceProcessor(OnnxProbabilityModel.fromAssetsOrNull(assets))
        sessionRepository = RoomSessionRepository(
            Room.databaseBuilder(applicationContext, AppDatabase::class.java, DATABASE_NAME).build(),
            onCommittedEpochs = { epochs ->
                epochs.forEach { epoch ->
                    liveInference.accept(epoch)?.let { prediction ->
                        sessionRepository.recordPrediction(
                            PredictionEventEntity(
                                sessionId = epoch.sessionId,
                                occurredAtEpochMillis = epoch.endEpochMillis,
                                probability = prediction.probability,
                                validInput = true,
                            ),
                        )
                    }
                }
            },
        )
        acknowledgementRetry = WatchAcknowledgementRetryCoordinator(
            pendingKeys = sessionRepository::pendingWatchAcknowledgements,
            send = { key -> PhoneBatchDataLayerRecovery.acknowledge(this@SmartSleepApplication, key) },
            markDelivered = sessionRepository::markWatchAcknowledged,
        )
        applicationScope.launch {
            try {
                acknowledgementRetry.ensureRetryLoop(applicationScope)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                Log.w(TAG, "Unable to start watch acknowledgement retry.", error)
            }
        }
        applicationScope.launch {
            WearableReceiver.receivedBatches.collect { delivery ->
                try {
                    sessionRepository.append(delivery)
                } catch (error: CancellationException) {
                    throw error
                } catch (error: Throwable) {
                    Log.w(TAG, "Unable to persist phone sensor batch.", error)
                }
                try {
                    // A failed release-journal confirmation can still leave a durable Room key.
                    acknowledgementRetry.ensureRetryLoop(applicationScope)
                } catch (error: CancellationException) {
                    throw error
                } catch (error: Throwable) {
                    Log.w(TAG, "Unable to wake watch acknowledgement retry.", error)
                }
            }
        }
        applicationScope.launch {
            try {
                PhoneBatchDataLayerRecovery.recover(this@SmartSleepApplication)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                Log.w(TAG, "Unable to replay persisted sensor batch Data Items.", error)
            }
        }
    }

    private companion object {
        const val TAG = "SmartSleepApplication"
        const val DATABASE_NAME = "smart-sleep.db"
    }
}
