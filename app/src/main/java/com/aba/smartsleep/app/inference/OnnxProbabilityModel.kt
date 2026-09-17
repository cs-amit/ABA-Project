package com.aba.smartsleep.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import com.aba.smartsleep.core.inference.ProbabilityModel
import java.nio.FloatBuffer

/** Optional ONNX Runtime bridge. Absence or load failure leaves inference disabled. */
class OnnxProbabilityModel private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
) : ProbabilityModel, AutoCloseable {
    override fun predict(shape: LongArray, values: FloatArray): Float {
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), shape).use { input ->
            session.run(mapOf(INPUT_NAME to input)).use { result ->
                val output = result.get(OUTPUT_NAME).orElseThrow().value
                return when (output) {
                    is FloatArray -> output.single()
                    is Array<*> -> (output.single() as FloatArray).single()
                    else -> error("Unexpected ONNX probability output")
                }
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val ASSET_PATH = "models/mesa_transfer.onnx"
        private const val INPUT_NAME = "sequences"
        private const val OUTPUT_NAME = "probability"

        fun fromAssetsOrNull(assets: AssetManager): ProbabilityModel? = OnnxProbabilityModelLoader.load(
            bytes = { runCatching { assets.open(ASSET_PATH).use { it.readBytes() } }.getOrNull() },
            create = { modelBytes ->
                val environment = OrtEnvironment.getEnvironment()
                OnnxProbabilityModel(environment, environment.createSession(modelBytes))
            },
        )
    }
}

internal object OnnxProbabilityModelLoader {
    fun load(bytes: () -> ByteArray?, create: (ByteArray) -> ProbabilityModel): ProbabilityModel? =
        runCatching { bytes()?.takeIf(ByteArray::isNotEmpty)?.let(create) }.getOrNull()
}
