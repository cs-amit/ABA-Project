package com.aba.smartsleep.core.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorQuality
import com.aba.smartsleep.core.model.SensorSample
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SensorBatchCodecTest {
    @Test
    fun `round trip preserves nullable IBI and timestamps`() {
        val original = SensorBatch(
            "s1",
            listOf(
                sampleAt(1_000, ibiMillis = null),
                sampleAt(1_040, ibiMillis = 812),
            ),
        )

        assertEquals(original, SensorBatchCodec.decode(SensorBatchCodec.encode(original)))
    }

    @Test
    fun `round trip preserves nullable sensor fields and quality`() {
        val original = SensorBatch(
            "s1",
            listOf(
                SensorSample(
                    timestampEpochMillis = 1_000,
                    accelX = 1.25f,
                    accelY = null,
                    accelZ = -3.5f,
                    heartRateBpm = 62.5f,
                    ibiMillis = null,
                    quality = SensorQuality.DEGRADED,
                ),
            ),
        )

        assertEquals(original, SensorBatchCodec.decode(SensorBatchCodec.encode(original)))
    }

    @Test
    fun `decode rejects unsupported payload version`() {
        assertFailsWith<IllegalArgumentException> { SensorBatchCodec.decode(byteArrayOf(99)) }
    }

    @Test
    fun `encode rejects samples spanning more than thirty seconds`() {
        val batch = SensorBatch("s1", listOf(sampleAt(1_000), sampleAt(31_001)))

        assertFailsWith<IllegalArgumentException> { SensorBatchCodec.encode(batch) }
    }

    @Test
    fun `decode rejects payloads at the ninety kilobyte limit`() {
        assertFailsWith<IllegalArgumentException> {
            SensorBatchCodec.decode(ByteArray(90 * 1024))
        }
    }

    @Test
    fun `decode rejects malformed nullable presence markers`() {
        assertFailsWith<IllegalArgumentException> {
            SensorBatchCodec.decode(payloadWithSingleSample { writeByte(2) })
        }
    }

    @Test
    fun `decode rejects malformed sensor quality`() {
        assertFailsWith<IllegalArgumentException> {
            SensorBatchCodec.decode(payloadWithSingleSample {
                repeat(5) { writeByte(0) }
                writeByte(99)
            })
        }
    }

    @Test
    fun `decode rejects trailing payload data`() {
        val encoded = SensorBatchCodec.encode(SensorBatch("s1", listOf(sampleAt(1_000))))

        assertFailsWith<IllegalArgumentException> { SensorBatchCodec.decode(encoded + byteArrayOf(0)) }
    }

    private fun sampleAt(timestampEpochMillis: Long, ibiMillis: Int? = null): SensorSample = SensorSample(
        timestampEpochMillis = timestampEpochMillis,
        accelX = null,
        accelY = null,
        accelZ = null,
        heartRateBpm = 60f,
        ibiMillis = ibiMillis,
        quality = SensorQuality.VALID,
    )

    private fun payloadWithSingleSample(sampleFields: DataOutputStream.() -> Unit): ByteArray =
        ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeByte(1)
                output.writeUTF("s1")
                output.writeInt(1)
                output.writeLong(1_000)
                output.sampleFields()
            }
            bytes.toByteArray()
        }
}
