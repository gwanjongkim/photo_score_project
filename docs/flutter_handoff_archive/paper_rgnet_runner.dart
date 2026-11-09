import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart';
import '../feature/a_cut/layer/inference/tflite_interpreter_manager.dart';
import '../utils/paper_rgnet_preprocessing.dart';

/// Runs the 'rgnet_paper_aadb_fp16' TFLite model.
///
/// ### Model contract
/// | Property | Value                                |
/// |----------|--------------------------------------|
/// | File     | rgnet_paper_aadb_fp16.tflite         |
/// | Input    | [1, 256, 256, 3] float32, RGB [0,255]|
/// | Output   | [1, 1] float32 scalar in [0, 1]      |
///
/// ### Built-in Preprocessing
/// This model includes a DenseNetV1UnitPreprocess layer that handles
/// scaling to [0, 1] and ImageNet mean/std subtraction.
class PaperRgnetRunner {
  PaperRgnetRunner({
    this.modelAssetPath = 'assets/models/rgnet_paper_aadb_fp16.tflite',
    TfliteInterpreterManager? interpreterManager,
  }) : _manager = interpreterManager ?? TfliteInterpreterManager.instance;

  final String modelAssetPath;
  final TfliteInterpreterManager _manager;
  bool _modelAvailable = true;

  bool get isAvailable => _modelAvailable;

  /// Score an image for aesthetic quality using the paper RGNet model.
  Future<Map<String, dynamic>?> scoreImage(Uint8List imageBytes) async {
    if (!_modelAvailable) return null;

    final stopwatch = Stopwatch()..start();
    try {
      // 1. Preprocess
      final preprocessStart = stopwatch.elapsedMilliseconds;
      final input = await preprocessForPaperRgnet(imageBytes);
      if (input == null) return null;
      final preprocessEnd = stopwatch.elapsedMilliseconds;

      // 2. Load Interpreter
      final interpreterStart = stopwatch.elapsedMilliseconds;
      final Interpreter interpreter;
      try {
        interpreter = await _manager.getInterpreter(
          modelAssetPath,
          useFlexDelegate: false, // Paper RGNet ops are standard
        );
      } catch (_) {
        _modelAvailable = false;
        return null;
      }
      final interpreterEnd = stopwatch.elapsedMilliseconds;

      // 3. Run Inference
      final inferenceStart = stopwatch.elapsedMilliseconds;
      final inputBytes = input.buffer.asUint8List();
      final outputElements = interpreter.getOutputTensor(0).numElements();
      final outputBytes = Uint8List(outputElements * 4);

      interpreter.run(inputBytes, outputBytes.buffer);

      final score = outputBytes.buffer.asFloat32List()[0];
      final inferenceEnd = stopwatch.elapsedMilliseconds;

      return {
        'score': score.clamp(0.0, 1.0),
        'latency_total': stopwatch.elapsedMilliseconds,
        'latency_preprocess': preprocessEnd - preprocessStart,
        'latency_load': interpreterEnd - interpreterStart,
        'latency_inference': inferenceEnd - inferenceStart,
      };
    } catch (_) {
      _modelAvailable = false;
      return null;
    } finally {
      stopwatch.stop();
    }
  }
}
