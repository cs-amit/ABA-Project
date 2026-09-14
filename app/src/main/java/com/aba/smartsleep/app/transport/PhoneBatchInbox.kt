package com.aba.smartsleep.app.transport

import com.aba.smartsleep.core.model.SensorBatch
import com.aba.smartsleep.core.transport.SensorBatchCodec
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Base64
import java.util.UUID
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

fun interface AtomicPayloadWriter {
    fun write(target: File, payload: ByteArray): Boolean
}

fun interface JournalRecordWriter {
    fun append(journal: File, record: String): Boolean
}

/**
 * Syncs the open journal file descriptor, but does not claim to sync its parent directory.
 * Initial filename recovery is supplied separately by [PhoneBatchStartupReplay].
 */
object FileContentSyncedJournalWriter : JournalRecordWriter {
    override fun append(journal: File, record: String): Boolean = try {
        val parent = requireNotNull(journal.parentFile)
        check(parent.exists() || parent.mkdirs()) { "Unable to create phone-private journal storage." }
        FileOutputStream(journal, true).use { output ->
            output.write((record + "\n").toByteArray(StandardCharsets.UTF_8))
            output.fd.sync()
        }
        true
    } catch (_: Exception) {
        false
    }
}

object AtomicFilePayloadWriter : AtomicPayloadWriter {
    override fun write(target: File, payload: ByteArray): Boolean {
        val parent = requireNotNull(target.parentFile)
        check(parent.exists() || parent.mkdirs()) { "Unable to create phone-private batch storage." }
        val temporary = File(parent, ".${target.name}.${UUID.randomUUID()}.tmp")
        return try {
            FileOutputStream(temporary).use { output ->
                output.write(payload)
                output.fd.sync()
            }
            try {
                Files.move(
                    temporary.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            true
        } catch (_: Exception) {
            temporary.delete()
            false
        }
    }
}

class PhoneBatchInbox(
    private val root: File,
    private val payloadWriter: AtomicPayloadWriter = AtomicFilePayloadWriter,
    private val journalWriter: JournalRecordWriter = FileContentSyncedJournalWriter,
) {
    fun accept(sessionId: String, sequence: Long, payload: ByteArray): Boolean {
        requireSafeSessionId(sessionId)
        require(sequence >= 0) { "Batch sequence must not be negative." }
        val state = journalState()
        val key = DeliveryKey(sessionId, sequence)
        state.accepted[key]?.let { existing -> return existing.contentEquals(payload) }
        if (!appendJournal(ACCEPTED, sessionId, sequence, payload)) return false
        val target = batchFile(sessionId, sequence)
        val parent = requireNotNull(target.parentFile)
        if (parent.exists() || parent.mkdirs()) {
            runCatching { payloadWriter.write(target, payload) }
        }
        return true
    }

    fun beginReleaseNext(): Delivery? {
        val state = journalState()
        state.accepted.keys.map(DeliveryKey::sessionId).distinct().sorted().forEach { sessionId ->
            var sequence = 0L
            while (DeliveryKey(sessionId, sequence) in state.completed) sequence += 1
            val key = DeliveryKey(sessionId, sequence)
            val payload = state.accepted[key] ?: return@forEach
            if (key !in state.leased && !appendJournal(LEASED, sessionId, sequence, null)) return null
            val batch = runCatching { SensorBatchCodec.decode(payload) }.getOrNull() ?: return null
            return Delivery(key, batch)
        }
        return null
    }

    fun completeRelease(delivery: Delivery): Boolean =
        journalState().let { state ->
            when {
                delivery.key in state.completed -> true
                delivery.key !in state.accepted -> false
                else -> appendJournal(COMPLETED, delivery.key.sessionId, delivery.key.sequence, null)
            }
        }

    fun isReleaseCompleted(delivery: Delivery): Boolean = delivery.key in journalState().completed

    private fun batchFile(sessionId: String, sequence: Long): File =
        File(sessionDirectory(sessionId), "$sequence.bin")

    private fun sessionDirectory(sessionId: String): File =
        File(root, Base64.getUrlEncoder().withoutPadding().encodeToString(sessionId.toByteArray(Charsets.UTF_8)))

    private fun appendJournal(action: String, sessionId: String, sequence: Long, payload: ByteArray?): Boolean {
        val encodedSession = Base64.getUrlEncoder().withoutPadding().encodeToString(sessionId.toByteArray())
        val encodedPayload = payload?.let { Base64.getUrlEncoder().withoutPadding().encodeToString(it) } ?: NO_PAYLOAD
        return journalWriter.append(File(root, JOURNAL_FILE), "$action\t$encodedSession\t$sequence\t$encodedPayload")
    }

    private fun journalState(): JournalState {
        val state = JournalState()
        File(root, JOURNAL_FILE).takeIf(File::isFile)?.forEachLine { line ->
            val fields = line.split('\t')
            if (fields.size != 4) return@forEachLine
            val sessionId = runCatching { String(Base64.getUrlDecoder().decode(fields[1])) }.getOrNull() ?: return@forEachLine
            val sequence = fields[2].toLongOrNull()?.takeIf { it >= 0 } ?: return@forEachLine
            val key = DeliveryKey(sessionId, sequence)
            when (fields[0]) {
                ACCEPTED -> runCatching { Base64.getUrlDecoder().decode(fields[3]) }.getOrNull()?.let {
                    state.accepted.putIfAbsent(key, it)
                }
                LEASED -> state.leased += key
                COMPLETED -> state.completed += key
            }
        }
        return state
    }

    data class DeliveryKey(val sessionId: String, val sequence: Long)

    data class Delivery(val key: DeliveryKey, val batch: SensorBatch)

    private class JournalState {
        val accepted = linkedMapOf<DeliveryKey, ByteArray>()
        val leased = mutableSetOf<DeliveryKey>()
        val completed = mutableSetOf<DeliveryKey>()
    }

    private companion object {
        const val JOURNAL_FILE = "accepted-batches.journal"
        const val ACCEPTED = "A"
        const val LEASED = "L"
        const val COMPLETED = "C"
        const val NO_PAYLOAD = "-"
    }
}

class PhoneBatchReleaseCoordinator(private val inbox: PhoneBatchInbox) {
    private val lock = Any()
    private val updates = Channel<Unit>(Channel.CONFLATED)
    private val retryRequested = mutableSetOf<PhoneBatchInbox.DeliveryKey>()

    fun accept(sessionId: String, sequence: Long, payload: ByteArray): Boolean = synchronized(lock) {
        inbox.accept(sessionId, sequence, payload).also { accepted ->
            if (accepted) updates.trySend(Unit)
        }
    }

    fun deliveries(): Flow<PhoneBatchDelivery> = flow {
        while (true) {
            val delivery = synchronized(lock) { inbox.beginReleaseNext() }
            if (delivery == null) {
                updates.receive()
            } else {
                emit(
                    PhoneBatchDelivery(
                        key = delivery.key,
                        batch = delivery.batch,
                        confirm = { confirmPersisted(delivery) },
                    ),
                )
                while (synchronized(lock) { !inbox.isReleaseCompleted(delivery) && delivery.key !in retryRequested }) {
                    updates.receive()
                }
                synchronized(lock) { retryRequested.remove(delivery.key) }
            }
        }
    }

    private fun confirmPersisted(delivery: PhoneBatchInbox.Delivery): Boolean = synchronized(lock) {
        inbox.completeRelease(delivery).also { completed ->
            if (!completed) retryRequested += delivery.key
            updates.trySend(Unit)
        }
    }
}

class PhoneBatchDelivery internal constructor(
    val key: PhoneBatchInbox.DeliveryKey,
    val batch: SensorBatch,
    private val confirm: () -> Boolean,
) {
    /**
     * Call only after the consumer's idempotent persistence transaction has committed.
     * A false result leaves this batch eligible for redelivery.
     */
    fun confirmPersisted(): Boolean = confirm()
}

data class PersistedBatchDataItem(
    val path: String,
    val payload: ByteArray,
)

class PhoneBatchStartupReplay(private val handler: PhoneDataLayerHandler) {
    fun recover(items: Iterable<PersistedBatchDataItem>): List<PhoneDataLayerHandler.AcknowledgementDecision> =
        items.mapNotNull { item -> handler.handle(item.path, item.payload) }
}

class PhoneDataLayerHandler(private val coordinator: PhoneBatchReleaseCoordinator) {
    fun handle(path: String, payload: ByteArray): AcknowledgementDecision? {
        val match = BATCH_PATH.matchEntire(path) ?: return null
        val sessionId = match.groupValues[1]
        val sequence = match.groupValues[2].toLongOrNull() ?: return null
        if (!isSafeSessionId(sessionId)) return null
        val batch = runCatching { SensorBatchCodec.decode(payload) }.getOrNull() ?: return null
        if (batch.sessionId != sessionId) return null
        if (!coordinator.accept(sessionId, sequence, payload)) return null
        return AcknowledgementDecision(sessionId, sequence)
    }

    data class AcknowledgementDecision(
        val sessionId: String,
        val sequence: Long,
        val shouldAcknowledge: Boolean = true,
    )

    private companion object {
        val BATCH_PATH = Regex("^/sessions/([^/]+)/batches/(0|[1-9][0-9]*)$")
    }
}

internal fun requireSafeSessionId(sessionId: String) {
    require(isSafeSessionId(sessionId)) { "Session IDs must be non-blank safe path segments." }
}

internal fun isSafeSessionId(sessionId: String): Boolean =
    sessionId.isNotBlank() && sessionId != "." && sessionId != ".." && '/' !in sessionId
