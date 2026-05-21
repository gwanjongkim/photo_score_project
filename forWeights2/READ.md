# forWeights2 실행 안내

## 1. 목적

`forWeights2`는 A-cut 캡스톤 프로젝트에서 로컬 이미지 폴더를 4개 미적 평가 모델로 채점하고, 브라우저에서 모델 가중치를 즉시 바꿔 final score와 ranking을 비교하는 독립 실험 도구입니다.

기존 `forWeights`는 NIMA, RGNet, A-LAMP 3개 모델 기반입니다. `forWeights2`는 그 구조와 실행 방식을 유지하면서 ICAA 모델을 추가해 4개 모델 기반으로 확장했습니다.

## 2. 디렉토리 구조

```text
forWeights2/
  configs/aesthetic_weight_lab.yaml
  models/aesthetic/
  test_images/
  tools/aesthetic_weight_lab/
  outputs/
  READ.md
  requirements.txt
```

- `configs/aesthetic_weight_lab.yaml`은 모델 경로, score key, input size, 기본 weight를 정의합니다.
- `models/aesthetic/`에는 `.tflite` 모델 파일을 둡니다.
- `test_images/`에는 평가할 이미지를 넣습니다.
- `tools/aesthetic_weight_lab/`에는 실행 코드와 HTML 리포트 생성 코드가 있습니다.
- `outputs/`에는 실행 결과가 생성됩니다.

## 3. 사용 모델

| 모델 | 파일명 | score key | type | score_index | 기본 weight |
|---|---|---|---|---|---|
| NIMA | `nima_mobile.tflite` | `nima_score` | `nima_distribution` | 해당 없음 | 0.25 |
| RGNet | `rgnet_paper_aadb_fp16.tflite` | `rgnet_score` | `scalar_tflite` | 해당 없음 | 0.25 |
| A-LAMP | `mobile_alamp_v2_fp16.tflite` | `alamp_score` | `alamp_signature` | 해당 없음 | 0.25 |
| ICAA | `icaa_dat_tf_native_fp16.tflite` | `icaa_score` | `vector_tflite` | 0 | 0.25 |

모델 파일은 모두 아래 위치에 있어야 합니다.

```text
forWeights2/models/aesthetic/
```

## 4. 이미지 넣는 방법

평가할 이미지를 아래 폴더에 복사합니다.

```text
forWeights2/test_images/
```

기본 지원 확장자는 `.jpg`, `.jpeg`, `.png`, `.webp`입니다. 개인 테스트 이미지는 커밋하지 마세요.

## 5. 가상환경 생성

Linux, WSL, macOS에서는 `forWeights2` 안에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell에서는 `forWeights2` 안에서 실행합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell에서 스크립트 실행 정책 오류가 나면 현재 셸에서만 허용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 6. 의존성 설치

기본 설치 명령은 다음과 같습니다.

```bash
pip install -r requirements.txt
```

runner는 `tflite_runtime`이 설치되어 있으면 먼저 사용하고, 없으면 TensorFlow의 `tf.lite.Interpreter`를 사용합니다. `tflite_runtime` 설치가 실패하면 TensorFlow 설치만으로도 실행할 수 있습니다.

## 7. 실행 명령

Linux, WSL, macOS에서는 다음 명령을 사용합니다.

```bash
python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/aesthetic_weight_lab_demo
```

Windows PowerShell에서는 줄바꿈 문자가 다릅니다.

```powershell
python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py `
  --input_dir test_images `
  --config configs/aesthetic_weight_lab.yaml `
  --output_dir outputs/aesthetic_weight_lab_demo
```

이미 출력 폴더가 있고 비어 있지 않으면 실행이 중단됩니다. 새 output directory를 지정하세요.

## 8. 결과 파일

실행 후 `outputs/aesthetic_weight_lab_demo/` 아래에 생성됩니다.

- `report.html`
- `raw_scores.csv`
- `raw_scores.json`
- `weight_presets.json`
- `command_log.txt`
- `thumbs/`
- `copied_images/`

## 9. HTML 리포트 사용

`outputs/aesthetic_weight_lab_demo/report.html`을 브라우저로 엽니다.

HTML에는 NIMA, RGNet, A-LAMP, ICAA 개별 점수와 final weighted score가 표시됩니다. 각 모델의 weight slider와 number input을 바꾸면 모델을 다시 실행하지 않고 브라우저 안에서 final score와 ranking이 즉시 갱신됩니다.

## 10. 최종 점수 계산식

각 이미지의 최종 점수는 활성 모델만 사용해 계산합니다.

```text
final_score = sum(weight_i * clamp(score_i, 0, 1)) / active_weight_sum
```

`active_weight_sum`은 점수가 존재하고 weight가 0보다 큰 모델들의 weight 합입니다. weight 합이 1이 아니어도 위 합으로 나누어 자동 정규화합니다. 어떤 모델이 실패해 점수가 `None`이면 그 모델은 계산에서 제외됩니다.

## 11. ICAA score_index

ICAA 모델은 scalar output 대신 `[MOS, Color]` 같은 벡터를 반환할 수 있습니다. `configs/aesthetic_weight_lab.yaml`의 ICAA 설정은 다음 값을 사용합니다.

```yaml
type: vector_tflite
score_index: 0
score_column: icaa_score
```

`score_index: 0`은 벡터의 첫 번째 값인 MOS를 `icaa_score`로 사용한다는 뜻입니다. output이 scalar이면 `score_index`가 0일 때만 그대로 사용합니다. output shape가 scalar, `[N]`, `[1,N]` 형태가 아니거나 `score_index`가 범위를 벗어나면 명확한 오류를 냅니다.

ICAA preprocessing은 export마다 다를 수 있습니다. 현재 config는 로컬 변환 보고서에서 확인된 224x224 RGB float 입력과 `zero_to_one` 정규화를 사용합니다. 추측으로 preprocessing을 바꾸지 말고, 새 모델을 교체할 때는 TFLite input details와 변환 보고서를 먼저 확인하세요.

## 12. 자주 발생하는 오류

### tflite_runtime 설치 실패

`tflite_runtime` wheel이 현재 Python 버전이나 OS를 지원하지 않을 수 있습니다. 이 경우 `tensorflow`를 설치하고 runner의 fallback을 사용하세요.

### tensorflow 설치 필요

`ModuleNotFoundError: No module named 'tensorflow'`가 나오면 가상환경을 활성화한 뒤 `pip install -r requirements.txt`를 다시 실행하세요.

### 모델 파일 없음

`Missing configured TFLite model file(s)` 오류가 나오면 4개 `.tflite` 파일이 `models/aesthetic/`에 있는지 확인하세요.

### 이미지 파일 없음

결과가 비어 있으면 `test_images/`에 지원 확장자의 이미지가 있는지 확인하세요. 하위 폴더까지 읽으려면 `--recursive`를 추가하세요.

### 입력 shape 또는 preprocess mismatch

TFLite input shape와 config의 `input_width`, `input_height`, `global_size`, `patch_size`, `patch_count`, `normalization`이 맞지 않으면 추론 오류가 나거나 점수 스케일이 왜곡될 수 있습니다. 특히 A-LAMP v2는 로컬 export metadata 기준으로 float pixel 0-255 입력을 사용합니다.

### output shape mismatch

`scalar_tflite output must be scalar` 또는 `vector_tflite output must be scalar, [N], or [1,N]` 오류가 나오면 해당 모델의 output details와 config type을 확인하세요. ICAA 벡터 모델은 `vector_tflite`와 `score_index`를 사용해야 합니다.

### 권한 문제

`outputs/` 또는 `test_images/`에 쓸 권한이 없으면 사용자 쓰기 권한이 있는 경로에서 실행하거나 output directory를 바꾸세요.

### Windows 경로 문제

PowerShell에서는 `/`와 `\`를 모두 어느 정도 처리하지만, 공백이 있는 경로는 따옴표로 감싸야 합니다.

```powershell
--input_dir "C:\Users\me\A cut images"
```

## 13. 검증 명령

문법 검사는 다음 명령으로 수행합니다.

```bash
python -m py_compile tools/aesthetic_weight_lab/*.py
```

설정 로드는 다음처럼 확인할 수 있습니다.

```bash
python - <<'PY'
from pathlib import Path
from tools.aesthetic_weight_lab.run_aesthetic_weight_lab import load_yaml_config
from tools.aesthetic_weight_lab.model_registry import enabled_model_specs, load_weights
root = Path.cwd()
config = load_yaml_config(root / "configs/aesthetic_weight_lab.yaml")
specs = enabled_model_specs(config, root)
weights = load_weights(config, specs)
print([(s.model_id, s.model_type, s.score_column, s.config.get("score_index")) for s in specs])
print(weights)
PY
```

모델 파일 체크섬은 다음 명령으로 확인할 수 있습니다.

```bash
sha256sum -c models/aesthetic/SHA256SUMS.txt
```

## 14. 주의사항

- 이 도구는 로컬 qualitative weight lab이며 공식 benchmark 재현이 아닙니다.
- HTML에서 weight를 바꾸는 동작은 이미 계산된 raw score를 재조합할 뿐입니다.
- 모델 파일, input shape, preprocessing을 바꾸면 Python scorer를 다시 실행해야 합니다.
- 모델별 score scale이 다를 수 있으므로 weight 비교 결과를 과학적 결론으로 과대 해석하지 마세요.
- 기존 `forWeights`는 수정하지 말고 `forWeights2` 안에서만 변경하세요.
