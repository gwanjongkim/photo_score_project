# ICAA QAT Feasibility Report

## Summary
- **Direct QAT possible**: False.
- **QAT can be applied safely in this workspace without new environment changes**: False.
- **Recommended decision from feasibility**: `E`.

## Confirmed Facts
- **SavedModel exists**: True at `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`.
- **`tf.saved_model.load` works**: True.
- **`tf.keras.models.load_model` works**: False.
- **Source model import works**: True.
- **Source model instantiate/build works**: True.
- **PyTorch checkpoint load into source model works**: True.
- **Source model kind**: `subclassed_keras_model`.
- **Source model has Functional inputs/outputs**: inputs=False, outputs=False.
- **`tf.keras.models.clone_model(source_model)` works**: False.
- **`tensorflow_model_optimization` available**: False.
- **`torch` available in audit interpreter**: True.

## Blockers
- SavedModel is not directly loadable as a trainable Keras model with tf.keras.models.load_model.
- Recoverable source model is subclassed with custom layers, not a Functional/Sequential graph model.
- Source model could not be cloned with tf.keras.models.clone_model, which standard TFMOT QAT commonly requires.
- tensorflow_model_optimization is not installed in the checked interpreters; installing it would modify the environment outside the allowed output directory.

## SavedModel Details
- **SavedModel probe**: `{'path': '/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model', 'exists': True, 'tf_saved_model_load': {'ok': True, 'object_type': '_UserObject', 'signatures': ['serving_default'], 'structured_input_signature': "((), {'image': TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='image')})", 'structured_outputs': "{'scores': TensorSpec(shape=(None, 2), dtype=tf.float32, name='scores')}"}, 'keras_load_model': {'ok': False, 'error_type': 'ValueError', 'error': "File format not supported: filepath=/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name)."}}`.

## Source Model Details
- **Source probe**: `{'checkpoint_path': '/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth', 'checkpoint_exists': True, 'source_root': '/home/omen_pc1/photo_score_project/experiments/icaa_tf_native', 'import_ok': True, 'instantiate_ok': True, 'load_pytorch_weights_ok': True, 'clone_model_ok': False, 'model_kind': 'subclassed_keras_model', 'is_keras_model': True, 'model_class': 'TFICAA17KDAT', 'has_functional_inputs': False, 'has_functional_outputs': False, 'layer_count': 13, 'layer_classes': ['Dense', 'LayerNormalization', 'TFDownProjection', 'TFPatchProjection', 'TFSoftHistogram', 'TFTransformerStage0', 'TFTransformerStage1', 'TFTransformerStage2', 'TFTransformerStage3'], 'state_dict_note': 'checkpoint object is an OrderedDict state_dict', 'state_dict_key_count': 487, 'clone_model_error_type': 'TypeError', 'clone_model_error': "<class 'tf_models.tf_dat_model.TFICAA17KDAT'> could not be deserialized properly. Please ensure that components that are Python object instances (layers, models, etc.) returned by `get_config()` are explicitly deserialized in the model's `from_config()` method.\n\nconfig={'module': 'tf_models.tf_dat_model', 'class_name': 'TFICAA17KDAT', 'config': {'name': 'tf_icaa17k_dat', 'epsilon': 1e-05, 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}}, 'registered_name': 'TFICAA17KDAT', 'compile_config': {}}.\n\nException encountered: Unable to revive model from config. When overriding the `get_config()` method, make sure that the returned config contains all items used as arguments in the  constructor to <class 'tf_models.tf_dat_model.TFICAA17KDAT'>, which is the default behavior. You can override this default behavior by defining a `from_config(cls, config)` class method to specify how to create an instance of TFICAA17KDAT from its config.\n\nReceived config={'name': 'tf_icaa17k_dat', 'epsilon': 1e-05, 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}}\n\nError encountered during deserialization: TFICAA17KDAT.__init__() got an unexpected keyword argument 'trainable'"}`.

## TFMOT Details
- **TFMOT probe**: `{'tfmot_available': False, 'quantize_model_available': False, 'quantize_apply_available': False, 'blocked_reason': 'tensorflow_model_optimization is not installed'}`.

## Conservative Conclusion
- The source model is recoverable as a subclassed Keras model from code plus the PyTorch checkpoint.
- The production SavedModel itself is a graph export and is not directly recoverable as a trainable Keras model with `tf.keras.models.load_model` in this environment.
- Direct, standard TFMOT QAT is blocked because the required package is absent and the model is a large subclassed/custom-layer DAT rather than a Functional/Sequential model that TFMOT can safely clone and quantize without custom configs.
- Creating manual fake-quant training or installing new packages would be a new unsupported experiment outside this conservative final attempt.
