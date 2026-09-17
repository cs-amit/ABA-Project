package com.aba.smartsleep.app.inference

import com.aba.smartsleep.core.inference.ProbabilityModel
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

class OnnxProbabilityModelLoaderTest {
    @Test
    fun `missing optional asset does not initialize runtime`() {
        var created = false

        val model = OnnxProbabilityModelLoader.load(bytes = { null }) {
            created = true
            ProbabilityModel { _, _ -> 0f }
        }

        assertNull(model)
        assert(!created)
    }

    @Test
    fun `available asset is passed to runtime factory`() {
        val expected = ProbabilityModel { _, _ -> 0.4f }
        val model = OnnxProbabilityModelLoader.load(bytes = { byteArrayOf(1) }) { expected }

        assertSame(expected, model)
    }

    @Test
    fun `runtime initialization failure leaves model unavailable`() {
        assertNull(OnnxProbabilityModelLoader.load(bytes = { byteArrayOf(1) }) { error("bad model") })
    }
}
