package com.example.gemmaharness

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.flow.collect
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import kotlin.system.measureTimeMillis

class BenchmarkRunner(
    private val context: Context,
    private val logger: (String) -> Unit
) {
    private var engine: Engine? = null
    private var modelLoaded: Boolean = false
    private var modelLoadTime: Long = 0
    private var memoryAfterLoadMb: Long = 0

    companion object {
        private const val TAG = "GemmaLatency"
        const val MODEL_PATH = "/data/local/tmp/llm/gemma4_e4b.litertlm"
        // Fallback path is not used by LiteRT-LM's EngineConfig, but we check it for logging.
        const val MODEL_NAME_FALLBACK = "gemma4_e4b.litertlm"
        const val BACKEND = "CPU" // LiteRT-LM manages its own delegates. This is for logging.
    }

    enum class PromptType(val key: String) {
        SELECTED_EXPLANATION("selected_explanation"),
        NEAR_MISS_FEEDBACK("near_miss_feedback"),
        REJECTION_REASON("rejection_reason")
    }

    private fun getPrompt(type: PromptType): String {
        // Prompts remain the same
        return when (type) {
            PromptType.SELECTED_EXPLANATION -> """{"selection_state": "selected", "comment_type": "selected_explanation"}"""
            PromptType.NEAR_MISS_FEEDBACK -> """{"selection_state": "near_miss", "comment_type": "near_miss_feedback"}"""
            PromptType.REJECTION_REASON -> """{"selection_state": "not_selected", "comment_type": "rejection_reason"}"""
        }
    }

    suspend fun loadModel() {
        if (modelLoaded) {
            logger("Model already loaded in ${modelLoadTime}ms.")
            return
        }

        logWithTag(Log.INFO, "LOAD_START", "--- Starting Model Load ---")
        preflightCheck()

        val modelFile = File(MODEL_PATH)
        if (!modelFile.exists()) {
            logWithTag(Log.ERROR, "LOAD_FAILED", "Model file not found at $MODEL_PATH")
            return
        }
        logWithTag(Log.INFO, "PREFLIGHT_OK", "Model file exists and is readable.")

        try {
            logWithTag(Log.INFO, "ENGINE_CONFIG_START", "Configuring LiteRT-LM Engine...")
            val engineConfig = EngineConfig(modelPath = MODEL_PATH)

            logWithTag(Log.INFO, "ENGINE_INITIALIZE_START", "Initializing LiteRT-LM Engine...")
            modelLoadTime = measureTimeMillis {
                engine = Engine(engineConfig).apply { initialize() }
            }
            logWithTag(Log.INFO, "ENGINE_INITIALIZE_SUCCESS", "Engine initialized successfully.")

            modelLoaded = true
            memoryAfterLoadMb = getAppMemoryUsage()
            logger("--- MODEL LOADED SUCCESSFULLY ---")
            logger("Model load time: ${modelLoadTime}ms")
            logger("Memory after load: ${memoryAfterLoadMb}MB")

        } catch (e: Exception) {
            logWithTag(Log.ERROR, "LOAD_FAILED", "Fatal error loading model: ${e.message}")
            val sw = StringWriter()
            e.printStackTrace(PrintWriter(sw))
            logger("STACK TRACE: 
$sw")
            modelLoaded = false
        }
    }

    suspend fun runSingle(type: PromptType, runId: String = "single") {
        if (!modelLoaded) {
            logger("Model not loaded. Please load the model first.")
            return
        }

        val prompt = getPrompt(type)
        logger("
--- Starting Run: $runId | Prompt: ${type.key} ---")

        val result = executeInference(prompt)

        val validation = JsonValidator.validate(result.output)
        val charsPerSec = if (result.totalGenerationTimeMs > 0) (result.output.length / (result.totalGenerationTimeMs / 1000.0)).toFloat() else 0.0f
        val memoryAfterGen = getAppMemoryUsage()

        logger("run_id: $runId")
        logger("prompt_type: ${type.key}")
        logger("backend: $BACKEND (via LiteRT-LM)")
        logger("model_load_time_ms: $modelLoadTime")
        logger("ttft_ms: ${result.ttftMs}")
        logger("total_generation_time_ms: ${result.totalGenerationTimeMs}")
        logger("output_char_count: ${result.output.length}")
        logger("chars_per_sec: $charsPerSec")
        logger("json_closed: ${validation.isClosed}")
        logger("json_parse_success: ${validation.parseSuccess}")
        logger("required_fields_present: ${validation.fieldsPresent}")
        logger("memory_after_load_mb: $memoryAfterLoadMb")
        logger("memory_after_generation_mb: $memoryAfterGen")
        logger("battery_temperature_before_c: ${result.tempBefore}")
        logger("battery_temperature_after_c: result.tempAfter")
        if (result.error != null) logger("error: ${result.error}")
        logger("--- Raw Output ---
${result.output}
--- End of Run ---")
    }

    private suspend fun executeInference(prompt: String): InferenceResult {
        var error: String? = null
        val output = StringBuilder()
        val tempBefore = getBatteryTemperature()
        var ttftMs = -1L
        val startTime = SystemClock.elapsedRealtimeNanos()

        try {
            logWithTag(Log.INFO, "CONVERSATION_START", "Creating conversation...")
            val conversation = engine!!.createConversation()
            logWithTag(Log.INFO, "GENERATION_START", "Sending message...")

            conversation.sendMessageAsync(prompt).collect { chunk ->
                if (ttftMs < 0 && chunk.text.isNotEmpty()) {
                    ttftMs = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000
                    logWithTag(Log.INFO, "FIRST_CHUNK", "First chunk received in ${ttftMs}ms.")
                }
                output.append(chunk.text)
            }
            logWithTag(Log.INFO, "GENERATION_SUCCESS", "Generation stream finished.")

        } catch (e: Exception) {
            logWithTag(Log.ERROR, "GENERATION_FAILED", "Error during generation: ${e.message}")
            error = e.message
            e.printStackTrace()
        }

        val endTime = SystemClock.elapsedRealtimeNanos()
        val tempAfter = getBatteryTemperature()
        return InferenceResult(
            output = output.toString(),
            ttftMs = ttftMs,
            totalGenerationTimeMs = (endTime - startTime) / 1_000_000,
            error = error,
            tempBefore = tempBefore,
            tempAfter = tempAfter
        )
    }

    suspend fun runAllOnce() {
        if (!modelLoaded) { loadModel() }
        if (!modelLoaded) return
        PromptType.values().forEachIndexed { i, type -> runSingle(type, "all_once_${i+1}") }
    }

    suspend fun runRepetitive(count: Int) {
        if (!modelLoaded) { loadModel() }
        if (!modelLoaded) return

        logger("
=== STARTING REPETITIVE BENCHMARK: $count runs per prompt ===")
        for (type in PromptType.values()) {
            val timings = mutableListOf<Long>()
            var successCount = 0
            for (i in 1..count) {
                val result = executeInference(getPrompt(type))
                if (result.error == null) {
                    timings.add(result.totalGenerationTimeMs)
                    if(JsonValidator.validate(result.output).parseSuccess) successCount++
                }
                runSingle(type, "repetitive_${type.key}_${i}_of_${count}")
            }
            if (timings.isNotEmpty()) logAggregates(type.key, timings, successCount, count)
        }
        logger("=== REPETITIVE BENCHMARK FINISHED ===")
    }

    private fun logAggregates(promptKey: String, timings: List<Long>, successCount: Int, totalRuns: Int) {
        logger("
--- Aggregates for '$promptKey' ($totalRuns runs) ---")
        logger("JSON Success: $successCount / $totalRuns")
        logger("Avg Time: ${timings.average().toLong()}ms")
        logger("Median Time: ${timings.sorted().getOrNull(timings.size / 2)}ms")
        logger("Min Time: ${timings.minOrNull()}ms")
        logger("Max Time: ${timings.maxOrNull()}ms")
    }

    private fun logWithTag(level: Int, tag: String, message: String) {
        logger("$tag: $message")
        Log.println(level, TAG, "$tag: $message")
    }

    private fun preflightCheck() {
        val primaryFile = File(MODEL_PATH)
        logWithTag(Log.INFO, "PREFLIGHT", "Checking primary path: ${primaryFile.absolutePath}")
        logFileStatus(primaryFile)

        val fallbackFile = File(context.filesDir, MODEL_NAME_FALLBACK)
        logWithTag(Log.INFO, "PREFLIGHT", "Checking fallback path for logging only: ${fallbackFile.absolutePath}")
        logFileStatus(fallbackFile)
    }

    private fun logFileStatus(file: File) {
        val exists = file.exists()
        logWithTag(Log.INFO, "FILE_STATUS", "  - Exists: $exists for ${file.name}")
        if (exists) {
            logWithTag(Log.INFO, "FILE_STATUS", "  - Can Read: ${file.canRead()}")
            logWithTag(Log.INFO, "FILE_STATUS", "  - Size: ${file.length()} bytes")
        }
    }

    private fun getAppMemoryUsage(): Long = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)

    private fun getBatteryTemperature(): Float {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)?.let { it / 10.0f } ?: -1.0f
    }

    private data class InferenceResult(
        val output: String,
        val ttftMs: Long,
        val totalGenerationTimeMs: Long,
        val error: String?,
        val tempBefore: Float,
        val tempAfter: Float
    )
}
