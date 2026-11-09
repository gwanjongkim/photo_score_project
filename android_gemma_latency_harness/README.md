# Gemma 4 E4B Android Latency Harness

This is a minimal Android application designed to measure the on-device inference latency of a Gemma 4 E4B model (`.litertlm`) on a target device like the Galaxy S23 Ultra.

## 1. Runtime Chosen

*   **Runtime:** LiteRT-LM Android (`com.google.ai.edge.litertlm:litertlm-android:0.2.0`)
*   **Reason for Change:** The previous runtime, MediaPipe LLM Inference, caused a fatal native crash (`SIGSEGV` in `LlmInferenceEngine_CreateSession`) when loading the `.litertlm` model. This indicated a fundamental incompatibility. The project has been switched to use the **LiteRT-LM** runtime, which is the correct, dedicated runtime for `.litertlm` model artifacts.

## 2. Model Setup

The application does not bundle the model. You must push it to the device using `adb`. The LiteRT-LM `EngineConfig` requires a direct file path, so only the primary path is actively used.

### Primary Path (Required)

```bash
# Create the directory on the device
adb shell mkdir -p /data/local/tmp/llm

# Push the model
# Replace <path_to_model>/gemma4_e4b.litertlm with the actual path on your computer
adb push <path_to_model>/gemma4_e4b.litertlm /data/local/tmp/llm/gemma4_e4b.litertlm
```

## 3. Build and Run

1.  **Open Project:** Open the `/home/omen_pc1/photo_score_project/android_gemma_latency_harness` directory in Android Studio.
2.  **SDK Location:** Android Studio will likely prompt you that the SDK location is not set. It will create a `local.properties` file for you. Allow it to do so.
3.  **Build:** Sync the Gradle project. It will download the new `litertlm-android` dependency.
4.  **Run:** Select a connected Galaxy S23 Ultra (or other target device) and run the `app` configuration.

## 4. How to Test

The UI and testing flow remain the same. The core change is the underlying inference engine.

*   **Load Model:** Initializes the LiteRT-LM engine with the model at `/data/local/tmp/llm/gemma4_e4b.litertlm`.
*   **Run Buttons:** Execute inference using the initialized engine.

## 5. Metrics and Data Collection

### In-App Metrics

The same metrics are collected. The app now also outputs verbose logs to Logcat with the tag `GemmaLatency` to trace the initialization and generation steps.

### Manual ADB Metrics (Recommended)

Use these `adb` commands for detailed analysis.

**Live Logcat view:**
```bash
adb logcat -s GemmaLatency
```

**Thermal Status:**
```bash
adb shell dumpsys thermalservice
```

**Battery Temperature:**
```bash
adb shell dumpsys battery | grep temperature
```

**Memory Usage:**
```bash
# Package name is com.example.gemmaharness
adb shell dumpsys meminfo com.example.gemmaharness
```

## 6. What to Send Back

After testing, please copy the entire text log from the app's screen. Also, provide the **full Logcat output** captured with `adb logcat -s GemmaLatency`, as it contains crucial step-by-step markers.
