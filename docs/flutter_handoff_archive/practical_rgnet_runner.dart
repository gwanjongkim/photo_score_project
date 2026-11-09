import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart';
import '../feature/a_cut/layer/inference/tflite_interpreter_manager.dart';
import '../utils/image_crop_utils.dart';
import 'package:flutter/material.dart' show Rect, Size;

/// Runs the 'rgnet_aadb_gpu' TFLite model.
///
/// ### Model contract
/// | Property | Value                              |
/// |----------|------------------------------------|
/// | File     | rgnet_aadb_gpu.tflite              |
/// | Input    | [1, 224, 224, 3] float32, RGB÷255  |
/// | Output   | [1, 1] float32 scalar in [0, 1]    |
class PracticalRgnetRunner {
  PracticalRgnetRunner({
    this.modelAssetPath = 'assets/models/rgnet_aadb_gpu.tflite',
    TfliteInterpreterManager? interpreterManager,
  }) : _manager = interpreterManager ?? TfliteInterpreterManager.instance;

  final String modelAssetPath;
  final TfliteInterpreterManager _manager;
  bool _modelAvailable = true;

  bool get isAvailable => _modelAvailable;

  Future<Map<String, dynamic>?> scoreImage(Uint8List imageBytes) async {
    if (!_modelAvailable) return null;

    final stopwatch = Stopwatch()..start();
    try {
      // 1. Preprocess (using existing utility, full image crop)
      final preprocessStart = stopwatch.elapsedMilliseconds;
      final input = await cropAndPreprocessForAadb(
        imageBytes,
        normalizedRect: const Rect.fromLTRB(0, 0, 1, 1),
        frameSize: Size.zero,
        targetSize: 224,
      );
      if (input == null) return null;
      final preprocessEnd = stopwatch.elapsedMilliseconds;

      // 2. Load Interpreter
      final interpreterStart = stopwatch.elapsedMilliseconds;
      final Interpreter interpreter;
      try {
        interpreter = await _manager.getInterpreter(
          modelAssetPath,
          useFlexDelegate: false,
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
