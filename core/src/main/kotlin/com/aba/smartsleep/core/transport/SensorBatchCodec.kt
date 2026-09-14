package com.aba.smartsleep.core.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.IOException

object SensorBatchCodec {
    private const val VERSION: Int = 1
    private const val MAX_BATCH_DURATION_MILLIS = 30_000L
    private const val MAX_PAYLOAD_BYTES = 90 * 1024
    private const val MIN_ENCODED_SAMPLE_BYTES = 13
    private const val PRESENT = 1
    private const val ABSENT = 0

    fun encode(batch: SensorBatch): ByteArray {
        validateBatchDuration(batch)
        val encoded = ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeByte(VERSION)
                output.writeUTF(batch.sessionId)
                output.writeInt(batch.samples.size)

                var previousTimestamp = 0L
                batch.samples.forEach { sample ->
                    output.writeLong(sample.timestampEpochMillis - previousTimestamp)
                    previousTimestamp = sample.timestampEpochMillis
                    output.writeOptionalFloat(sample.accelX)
                    output.writeOptionalFloat(sample.accelY)
                    output.writeOptionalFloat(sample.accelZ)
                    output.writeOptionalFloat(sample.heartRateBpm)
                    output.writeOptionalInt(sample.ibiMillis)
                    output.writeByte(sample.quality.ordinal)
                }
            }
            bytes.toByteArray()
        }
        require(encoded.size < MAX_PAYLOAD_BYTES) {
            "Encoded sensor batch must be smaller than $MAX_PAYLOAD_BYTES bytes."
        }
        return encoded
    }

    fun decode(payload: ByteArray): SensorBatch {
        require(payload.size < MAX_PAYLOAD_BYTES) {
            "Sensor batch payload must be smaller than $MAX_PAYLOAD_BYTES bytes."
        }
        try {
            return DataInputStream(ByteArrayInputStream(payload)).use { input ->
                require(input.readUnsignedByte() == VERSION) { "Unsupported sensor batch payload version." }
                val sessionId = input.readUTF()
                val sampleCount = input.readInt()
                require(sampleCount in 0..(MAX_PAYLOAD_BYTES / MIN_ENCODED_SAMPLE_BYTES)) {
                    "Invalid sensor batch sample count."
                }

                var previousTimestamp = 0L
                val samples = ArrayList<SensorSample>(sampleCount)
                repeat(sampleCount) {
                    val timestamp = Math.addExact(previousTimestamp, input.readLong())
                    previousTimestamp = timestamp
                    samples += SensorSample(
                        timestampEpochMillis = timestamp,
                        accelX = input.readOptionalFloat(),
                        accelY = input.readOptionalFloat(),
                        accelZ = input.readOptionalFloat(),
                        heartRateBpm = input.readOptionalFloat(),
                        ibiMillis = input.readOptionalInt(),
                        quality = input.readQuality(),
                    )
                }
                require(input.available() == 0) { "Unexpected trailing sensor batch payload data." }
                SensorBatch(sessionId, samples).also(::validateBatchDuration)
            }
        } catch (error: EOFException) {
            throw IllegalArgumentException("Truncated sensor batch payload.", error)
        } catch (error: ArithmeticException) {
            throw IllegalArgumentException("Invalid sensor batch timestamp delta.", error)
        } catch (error: IOException) {
            throw IllegalArgumentException("Unreadable sensor batch payload.", error)
        }
    }

    private fun validateBatchDuration(batch: SensorBatch) {
        val firstTimestamp = batch.samples.firstOrNull()?.timestampEpochMillis ?: return
        val lastTimestamp = batch.samples.last().timestampEpochMillis
        require(lastTimestamp - firstTimestamp <= MAX_BATCH_DURATION_MILLIS) {
            "Sensor batches may span at most $MAX_BATCH_DURATION_MILLIS milliseconds."
        }
    }

    private fun DataOutputStream.writeOptionalFloat(value: Float?) {
        writeByte(if (value == null) ABSENT else PRESENT)
        if (value != null) writeFloat(value)
    }

    private fun DataOutputStream.writeOptionalInt(value: Int?) {
        writeByte(if (value == null) ABSENT else PRESENT)
        if (value != null) writeInt(value)
    }

    private fun DataInputStream.readOptionalFloat(): Float? = when (val presence = readUnsignedByte()) {
        ABSENT -> null
        PRESENT -> readFloat()
        else -> throw IllegalArgumentException("Invalid float presence marker: $presence")
    }

    private fun DataInputStream.readOptionalInt(): Int? = when (val presence = readUnsignedByte()) {
        ABSENT -> null
        PRESENT -> readInt()
        else -> throw IllegalArgumentException("Invalid integer presence marker: $presence")
    }

    private fun DataInputStream.readQuality(): SensorQuality {
        val ordinal = readUnsignedByte()
        return SensorQuality.entries.getOrNull(ordinal)
            ?: throw IllegalArgumentException("Invalid sensor quality ordinal: $ordinal")
    }
}
