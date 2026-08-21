package com.aba.smartsleep.core.model

import kotlinx.coroutines.flow.Flow

interface WearableDataSource {
    suspend fun capabilities(): Set<WearableCapability>

    fun samples(): Flow<SensorSample>

    suspend fun start(sessionId: String)

    suspend fun stop()
}
