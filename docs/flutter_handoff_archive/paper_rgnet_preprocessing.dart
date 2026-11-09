import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// Prepares image input for the Paper RGNet AADB model.
/// 
/// Model Contract:
/// - Input Shape: [1, 256, 256, 3]
/// - DType: float32
/// - Normalization: None (built-in DenseNet preprocessing)
/// - Range: [0, 255] (internally scaled to [0, 1] then ImageNet mean/std applied)
/// 
/// This function resizes to 256x256 and converts to Float32List.
Future<Float32List?> preprocessForPaperRgnet(
  Uint8List imageBytes, {
  int targetSize = 256,
}) {
  return compute(
    _preprocessInIsolate,
    _PreprocessRequest(
      imageBytes: imageBytes,
      targetSize: targetSize,
    ),
  );
}

class _PreprocessRequest {
  final Uint8List imageBytes;
  final int targetSize;

  const _PreprocessRequest({
    required this.imageBytes,
    required this.targetSize,
  });
}

Float32List? _preprocessInIsolate(_PreprocessRequest req) {
  final decoded = img.decodeImage(req.imageBytes);
  if (decoded == null) return null;

  final resized = img.copyResize(
    decoded,
    width: req.targetSize,
    height: req.targetSize,
    interpolation: img.Interpolation.linear,
  );

  final total = req.targetSize * req.targetSize * 3;
  final output = Float32List(total);
  var cursor = 0;

  for (var y = 0; y < req.targetSize; y++) {
    for (var x = 0; x < req.targetSize; x++) {
      final pixel = resized.getPixel(x, y);
      
      // No external normalization: model expects [0, 255]
      // Built-in DenseNet preprocessing handles the rest.
      output[cursor++] = pixel.r.toDouble();
      output[cursor++] = pixel.g.toDouble();
      output[cursor++] = pixel.b.toDouble();
    }
  }

  return output;
}
