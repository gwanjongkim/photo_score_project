import 'dart:io';
import 'dart:typed_data';
import 'services/paper_rgnet_runner.dart';
import 'services/practical_rgnet_runner.dart';

// Note: This is a pseudo-test intended to be run in a Flutter environment.
// Since we are in a CLI without a Flutter runner, we will simulate the image loading
// and report the logic. In a real environment, this would be a widget or integration test.

void main() async {
  print('Starting RGNet Smoke/Parity Test...');

  final paperRunner = PaperRgnetRunner();
  final practicalRunner = PracticalRgnetRunner();

  final testImages = [
    'farm1_319_19982213545_f2c6a56ea7_b.jpg',
    'farm1_316_20145499186_18fcea17fd_b.jpg',
    'farm1_292_20177190071_ed73bfdf45_b.jpg',
    'farm4_3747_20150487562_e5683def63_b.jpg',
    'farm1_295_19542413113_995218a1dc_b.jpg',
  ];

  // Reference values from predictions.csv
  final expectedFp16Preds = {
    'farm1_319_19982213545_f2c6a56ea7_b.jpg': 0.3016504943370819,
    'farm1_316_20145499186_18fcea17fd_b.jpg': 0.6825769543647766,
    'farm1_292_20177190071_ed73bfdf45_b.jpg': 0.40095025300979614,
    'farm4_3747_20150487562_e5683def63_b.jpg': 0.45926937460899353,
    'farm1_295_19542413113_995218a1dc_b.jpg': 0.3152311146259308,
  };

  print('| Image | Python FP16 | Flutter Paper | Diff | Practical | Paper Latency | Pract Latency |');
  print('|-------|-------------|---------------|------|-----------|---------------|---------------|');

  for (final fileName in testImages) {
    final filePath = 'assets/test_images/rgnet_parity/$fileName';
    final file = File(filePath);
    if (!file.existsSync()) {
      print('Error: File not found $filePath');
      continue;
    }

    final bytes = file.readAsBytesSync();

    // 1. Run Paper RGNet
    final paperResult = await paperRunner.scoreImage(bytes);
    
    // 2. Run Practical RGNet
    final practicalResult = await practicalRunner.scoreImage(bytes);

    if (paperResult != null && practicalResult != null) {
      final pythonPred = expectedFp16Preds[fileName]!;
      final flutterPaperPred = paperResult['score'] as double;
      final diff = (flutterPaperPred - pythonPred).abs();
      final practicalPred = practicalResult['score'] as double;
      
      print('| $fileName | ${pythonPred.toStringAsFixed(4)} | ${flutterPaperPred.toStringAsFixed(4)} | ${diff.toStringAsFixed(6)} | ${practicalPred.toStringAsFixed(4)} | ${paperResult['latency_inference']}ms | ${practicalResult['latency_inference']}ms |');
    } else {
      print('| $fileName | Error running models |');
    }
  }
}
