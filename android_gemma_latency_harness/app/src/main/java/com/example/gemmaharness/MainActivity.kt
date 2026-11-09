package com.example.gemmaharness

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.lifecycle.lifecycleScope
import com.example.gemmaharness.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var benchmarkRunner: BenchmarkRunner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        benchmarkRunner = BenchmarkRunner(this) { message ->
            runOnUiThread {
                binding.logTextView.append("$message
")
            }
        }

        setupClickListeners()
        log("App initialized. Push model via ADB and press 'Load Model'.")
    }

    private fun setupClickListeners() {
        binding.buttonLoadModel.setOnClickListener { runBenchmark { benchmarkRunner.loadModel() } }
        binding.buttonRunSelected.setOnClickListener { runBenchmark { benchmarkRunner.runSingle(BenchmarkRunner.PromptType.SELECTED_EXPLANATION) } }
        binding.buttonRunNearMiss.setOnClickListener { runBenchmark { benchmarkRunner.runSingle(BenchmarkRunner.PromptType.NEAR_MISS_FEEDBACK) } }
        binding.buttonRunRejection.setOnClickListener { runBenchmark { benchmarkRunner.runSingle(BenchmarkRunner.PromptType.REJECTION_REASON) } }
        binding.buttonRunAllOnce.setOnClickListener { runBenchmark { benchmarkRunner.runAllOnce() } }
        binding.buttonRun5.setOnClickListener { runBenchmark { benchmarkRunner.runRepetitive(5) } }
        binding.buttonRun10.setOnClickListener { runBenchmark { benchmarkRunner.runRepetitive(10) } }
        binding.buttonClear.setOnClickListener { binding.logTextView.text = "" }
    }

    private fun runBenchmark(action: suspend () -> Unit) {
        setButtonsEnabled(false)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                action()
            } catch (e: Exception) {
                log("An error occurred: ${e.message}")
                e.printStackTrace()
            } finally {
                withContext(Dispatchers.Main) {
                    setButtonsEnabled(true)
                }
            }
        }
    }

    private fun setButtonsEnabled(isEnabled: Boolean) {
        binding.buttonsLayout.run {
            for (i in 0 until childCount) {
                getChildAt(i).isEnabled = isEnabled
            }
        }
    }

    private fun log(message: String) {
        runOnUiThread {
            binding.logTextView.append("$message
")
        }
    }
}
