package com.aba.smartsleep.wear.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.model.SensorSample
import com.aba.smartsleep.core.transport.SensorBatchCodec
import java.io.File
import java.util.Base64
import kotlinx.coroutines.CancellationException

internal class SensorBatchAccumulator(private val sessionId: String) {
    private val pending = mutableListOf<SensorSample>()

    fun append(sample: SensorSample): List<SensorBatch> {
        val candidate = (pending + sample).sortedBy(SensorSample::timestampEpochMillis)
        if (candidate.size == 1 || fitsBatch(candidate)) {
            pending.clear()
            pending.addAll(candidate)
            return emptyList()
        }

        val completed = SensorBatch(sessionId, pending.toList())
        pending.clear()
        pending += sample
        return listOf(completed)
    }

    fun flush(): SensorBatch? = pending.takeIf(List<SensorSample>::isNotEmpty)?.let { samples ->
        SensorBatch(sessionId, samples.toList()).also { pending.clear() }
    }

    private fun fitsBatch(samples: List<SensorSample>): Boolean {
        val duration = samples.last().timestampEpochMillis - samples.first().timestampEpochMillis
        if (duration > MAX_BATCH_DURATION_MILLIS) return false
        return runCatching { SensorBatchCodec.encode(SensorBatch(sessionId, samples)) }.isSuccess
    }

    private companion object {
        const val MAX_BATCH_DURATION_MILLIS = 30_000L
    }
}

internal class ResilientBatchTransfer(
    private val transfer: suspend (SensorBatch) -> Unit,
) {
    suspend fun transfer(batches: List<SensorBatch>): List<Throwable> = buildList {
        batches.forEach { batch ->
            try {
                transfer(batch)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                add(error)
            }
        }
    }
}

internal class PendingBatchFiles(private val root: File) {
    fun write(sessionId: String, sequence: Long, payload: ByteArray) {
        requireSafeWatchSessionId(sessionId)
        val sessionDirectory = sessionDirectory(sessionId)
        check(sessionDirectory.exists() || sessionDirectory.mkdirs()) {
            "Unable to create watch-private pending batch storage."
        }
        pendingFile(sessionId, sequence).outputStream().use { output -> output.write(payload) }
    }

    fun acknowledge(sessionId: String, sequence: Long) {
        requireSafeWatchSessionId(sessionId)
        val sessionDirectory = sessionDirectory(sessionId)
        check(sessionDirectory.exists() || sessionDirectory.mkdirs()) {
            "Unable to create watch-private acknowledgement storage."
        }
        acknowledgementFile(sessionId, sequence).outputStream().use { output -> output.write(1) }
        pendingFile(sessionId, sequence).delete()
    }

    fun isAcknowledged(sessionId: String, sequence: Long): Boolean {
        requireSafeWatchSessionId(sessionId)
        return acknowledgementFile(sessionId, sequence).isFile
    }

    fun pendingSequences(sessionId: String): List<Long> {
        requireSafeWatchSessionId(sessionId)
        return sessionDirectory(sessionId)
            .listFiles()
            .orEmpty()
            .mapNotNull { file -> file.takeIf(File::isFile)?.takeIf { it.extension == "bin" } }
            .mapNotNull { file -> file.nameWithoutExtension.toLongOrNull() }
            .filter { sequence ->
                if (isAcknowledged(sessionId, sequence)) {
                    pendingFile(sessionId, sequence).delete()
                    false
                } else {
                    true
                }
            }
            .sorted()
    }

    fun pendingFile(sessionId: String, sequence: Long): File {
        requireSafeWatchSessionId(sessionId)
        return File(sessionDirectory(sessionId), "$sequence.bin")
    }

    private fun acknowledgementFile(sessionId: String, sequence: Long): File =
        File(sessionDirectory(sessionId), "$sequence.ack")

    private fun sessionDirectory(sessionId: String): File =
        File(root, Base64.getUrlEncoder().withoutPadding().encodeToString(sessionId.toByteArray(Charsets.UTF_8)))
}

internal class PendingBatchResender(
    private val pendingFiles: PendingBatchFiles,
    private val upload: suspend (String, Long, ByteArray) -> Unit,
) {
    suspend fun resend(sessionId: String): List<Throwable> = buildList {
        pendingFiles.pendingSequences(sessionId).forEach { sequence ->
            try {
                upload(sessionId, sequence, pendingFiles.pendingFile(sessionId, sequence).readBytes())
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                add(error)
            }
        }
    }
}

internal object WatchAcknowledgementHandler {
    private val acknowledgementPath = Regex("^/sessions/([^/]+)/acknowledgements/(0|[1-9][0-9]*)$")

    fun handle(path: String, pendingFiles: PendingBatchFiles): Boolean {
        val match = acknowledgementPath.matchEntire(path) ?: return false
        val sessionId = match.groupValues[1]
        val sequence = match.groupValues[2].toLongOrNull() ?: return false
        if (!isSafeWatchSessionId(sessionId)) return false
        return runCatching {
            pendingFiles.acknowledge(sessionId, sequence)
            true
        }.getOrDefault(false)
    }
}

internal fun requireSafeWatchSessionId(sessionId: String) {
    require(isSafeWatchSessionId(sessionId)) {
        "Session IDs must be non-blank safe path segments."
    }
}

internal fun isSafeWatchSessionId(sessionId: String): Boolean =
    sessionId.isNotBlank() && sessionId != "." && sessionId != ".." && '/' !in sessionId
