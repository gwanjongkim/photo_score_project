# Repository snapshot

Captured at 2026-07-11T14:51:34+09:00 from `/home/omen_pc1/photo_score_project`.

## Git identity

- Commit: `879999b5510c2dfc32497279cac35a1ff6d601bc`.
- Branch: `main`.
- Upstream state at capture: `main...origin/main [ahead 10]`.
- Working tree was already dirty before this audit. Existing changes were treated as user-owned and left untouched.
- Porcelain totals at this capture: 14 tracked modified/status entries and 8965 untracked file entries. The audit directory itself appears among the untracked entries because this phase creates new evidence files there.

```text
commit 879999b5510c2dfc32497279cac35a1ff6d601bc
Author:     gwanjongkim <kimkanjong@gmail.com>
AuthorDate: Thu Jun 4 00:16:48 2026 +0900
Commit:     gwanjongkim <kimkanjong@gmail.com>
CommitDate: Thu Jun 4 00:16:48 2026 +0900

    feat(technical-iqa): add student manifest
```

## Tracked modified files

```text
M	.gitignore
M	README.md
M	configs/paper_benchmarks/alamp_multipatch_teacher_ava.yaml
M	forWeights2/HANDOFF.md
M	forWeights2/READ.md
M	forWeights2/checklist.md
M	forWeights2/configs/aesthetic_weight_lab.yaml
M	forWeights2/configs/aesthetic_weight_lab_alamp_finetuned.yaml
M	forWeights2/context-notes.md
M	forWeights2/models/aesthetic/MODEL_CARD.md
M	forWeights2/models/aesthetic/SHA256SUMS.txt
M	src/eval/evaluate_alamp_multipatch_teacher.py
M	src/train/train_alamp_multipatch_teacher.py
M	src/train/train_techiqa_guard.py
```

## Complete porcelain status including untracked files

The following is the literal output of `git status --porcelain=v1 --untracked-files=all` at capture time.

```text
Warning: truncated output (original token count: 807063)
Total output lines: 31121

 M .gitignore
 M README.md
 M configs/paper_benchmarks/alamp_multipatch_teacher_ava.yaml
 M forWeights2/HANDOFF.md
 M forWeights2/READ.md
 M forWeights2/checklist.md
 M forWeights2/configs/aesthetic_weight_lab.yaml
 M forWeights2/configs/aesthetic_weight_lab_alamp_finetuned.yaml
 M forWeights2/context-notes.md
 M forWeights2/models/aesthetic/MODEL_CARD.md
 M forWeights2/models/aesthetic/SHA256SUMS.txt
 M src/eval/evaluate_alamp_multipatch_teacher.py
 M src/train/train_alamp_multipatch_teacher.py
 M src/train/train_techiqa_guard.py
?? .venv_mdiqa/bin/Activate.ps1
?? .venv_mdiqa/bin/accelerate
?? .venv_mdiqa/bin/accelerate-config
?? .venv_mdiqa/bin/accelerate-estimate-memory
?? .venv_mdiqa/bin/accelerate-launch
?? .venv_mdiqa/bin/accelerate-merge-weights
?? .venv_mdiqa/bin/activate
?? .venv_mdiqa/bin/activate.csh
?? .venv_mdiqa/bin/activate.fish
?? .venv_mdiqa/bin/datasets-cli
?? .venv_mdiqa/bin/f2py
?? .venv_mdiqa/bin/fonttools
?? .venv_mdiqa/bin/ftfy
?? .venv_mdiqa/bin/futurize
?? .venv_mdiqa/bin/get_gprof
?? .venv_mdiqa/bin/get_objgraph
?? .venv_mdiqa/bin/hf
?? .venv_mdiqa/bin/httpx
?? .venv_mdiqa/bin/huggingface-cli
?? .venv_mdiqa/bin/identify-cli
?? .venv_mdiqa/bin/idna
?? .venv_mdiqa/bin/imageio_download_bin
?? .venv_mdiqa/bin/imageio_remove_bin
?? .venv_mdiqa/bin/isympy
?? .venv_mdiqa/bin/lsm2bin
?? .venv_mdiqa/bin/markdown_py
?? .venv_mdiqa/bin/nodeenv
?? .venv_mdiqa/bin/normalizer
?? .venv_mdiqa/bin/numba
?? .venv_mdiqa/bin/numpy-config
?? .venv_mdiqa/bin/pasteurize
?? .venv_mdiqa/bin/pip
?? .venv_mdiqa/bin/pip3
?? .venv_mdiqa/bin/pip3.12
?? .venv_mdiqa/bin/pre-commit
?? .venv_mdiqa/bin/proton
?? .venv_mdiqa/bin/proton-viewer
?? .venv_mdiqa/bin/py.test
?? .venv_mdiqa/bin/pyftmerge
?? .venv_mdiqa/bin/pyftsubset
?? .venv_mdiqa/bin/pygmentize
?? .venv_mdiqa/bin/pyiqa
?? .venv_mdiqa/bin/pytest
?? .venv_mdiqa/bin/python
?? .venv_mdiqa/bin/python3
?? .venv_mdiqa/bin/python3.12
?? .venv_mdiqa/bin/ruff
?? .venv_mdiqa/bin/tensorboard
?? .venv_mdiqa/bin/tiff2fsspec
?? .venv_mdiqa/bin/tiffcomment
?? .venv_mdiqa/bin/tifffile
?? .venv_mdiqa/bin/tiny-agents
?? .venv_mdiqa/bin/torchfrtrace
?? .venv_mdiqa/bin/torchrun
?? .venv_mdiqa/bin/tqdm
?? .venv_mdiqa/bin/transformers-cli
?? .venv_mdiqa/bin/ttx
?? .venv_mdiqa/bin/undill
?? .venv_mdiqa/bin/virtualenv
?? .venv_mdiqa/bin/yapf
?? .venv_mdiqa/bin/yapf-diff
?? .venv_mdiqa/lib64
?? .venv_mdiqa/pyvenv.cfg
?? .venv_mdiqa/share/man/man1/isympy.1
?? .venv_mdiqa/share/man/man1/ttx.1
?? .venv_tf/bin/Activate.ps1
?? .venv_tf/bin/activate
?? .venv_tf/bin/activate.csh
?? .venv_tf/bin/activate.fish
?? .venv_tf/bin/f2py
?? .venv_tf/bin/idna
?? .venv_tf/bin/markdown-it
?? .venv_tf/bin/normalizer
?? .venv_tf/bin/numpy-config
?? .venv_tf/bin/pip
?? .venv_tf/bin/pip3
?? .venv_tf/bin/pip3.12
?? .venv_tf/bin/pygmentize
?? .venv_tf/bin/python
?? .venv_tf/bin/python3
?? .venv_tf/bin/python3.12
?? .venv_tf/bin/saved_model_cli
?? .venv_tf/bin/tf_upgrade_v2
?? .venv_tf/bin/tflite_convert
?? .venv_tf/bin/toco
?? .venv_tf/bin/tqdm
?? .venv_tf/bin/wheel
?? .venv_tf/lib/python3.12/site-packages/81d243bd2c585b0f4821__mypyc.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/AvifImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/BdfFontFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/BlpImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/BmpImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/BufrStubImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ContainerIO.py
?? .venv_tf/lib/python3.12/site-packages/PIL/CurImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/DcxImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/DdsImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/EpsImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ExifTags.py
?? .venv_tf/lib/python3.12/site-packages/PIL/FitsImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/FliImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/FontFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/FpxImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/FtexImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GbrImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GdImageFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GifImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GimpGradientFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GimpPaletteFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/GribStubImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/Hdf5StubImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/IcnsImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/IcoImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/Image.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageChops.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageCms.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageColor.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageDraw.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageDraw2.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageEnhance.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageFilter.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageFont.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageGrab.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageMath.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageMode.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageMorph.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageOps.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImagePalette.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImagePath.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageQt.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageSequence.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageShow.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageStat.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageText.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageTk.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageTransform.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImageWin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/ImtImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/IptcImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/Jpeg2KImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/JpegImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/JpegPresets.py
?? .venv_tf/lib/python3.12/site-packages/PIL/McIdasImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/MicImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/MpegImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/MpoImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/MspImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PSDraw.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PaletteFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PalmImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PcdImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PcfFontFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PcxImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PdfImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PdfParser.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PixarImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PngImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PpmImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/PsdImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/QoiImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/SgiImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/SpiderImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/SunImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/TarIO.py
?? .venv_tf/lib/python3.12/site-packages/PIL/TgaImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/TiffImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/TiffTags.py
?? .venv_tf/lib/python3.12/site-packages/PIL/WalImageFile.py
?? .venv_tf/lib/python3.12/site-packages/PIL/WebPImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/WmfImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/XVThumbImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/XbmImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/XpmImagePlugin.py
?? .venv_tf/lib/python3.12/site-packages/PIL/__init__.py
?? .venv_tf/lib/python3.12/site-packages/PIL/__main__.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_avif.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_avif.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_binary.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_deprecate.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_imaging.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imaging.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingcms.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingcms.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingft.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingft.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingmath.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingmath.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingmorph.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingmorph.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingtk.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_imagingtk.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/_tkinter_finder.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_typing.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_util.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_version.py
?? .venv_tf/lib/python3.12/site-packages/PIL/_webp.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/PIL/_webp.pyi
?? .venv_tf/lib/python3.12/site-packages/PIL/features.py
?? .venv_tf/lib/python3.12/site-packages/PIL/py.typed
?? .venv_tf/lib/python3.12/site-packages/PIL/report.py
?? .venv_tf/lib/python3.12/site-packages/_distutils_hack/__init__.py
?? .venv_tf/lib/python3.12/site-packages/_distutils_hack/override.py
?? .venv_tf/lib/python3.12/site-packages/_yaml/__init__.py
?? .venv_tf/lib/python3.12/site-packages/absl/__init__.py
?? .venv_tf/lib/python3.12/site-packages/absl/app.py
?? .venv_tf/lib/python3.12/site-packages/absl/app.pyi
?? .venv_tf/lib/python3.12/site-packages/absl/command_name.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/__init__.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_argument_parser.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_defines.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_exceptions.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_flag.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_flagvalues.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_helpers.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_validators.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/_validators_classes.py
?? .venv_tf/lib/python3.12/site-packages/absl/flags/argparse_flags.py
?? .venv_tf/lib/python3.12/site-packages/absl/logging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/absl/logging/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/absl/logging/converter.py
?? .venv_tf/lib/python3.12/site-packages/absl/py.typed
?? .venv_tf/lib/python3.12/site-packages/absl/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/_bazelize_command.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/_pretty_print_reporter.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/absltest.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/flagsaver.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/parameterized.py
?? .venv_tf/lib/python3.12/site-packages/absl/testing/xml_reporter.py
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/licenses/AUTHORS
?? .venv_tf/lib/python3.12/site-packages/absl_py-2.4.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/AUTHORS.rst
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/LICENSE
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/astunparse-1.6.3.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/astunparse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/astunparse/__main__.py
?? .venv_tf/lib/python3.12/site-packages/astunparse/printer.py
?? .venv_tf/lib/python3.12/site-packages/astunparse/unparser.py
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/certifi-2026.5.20.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/certifi/__init__.py
?? .venv_tf/lib/python3.12/site-packages/certifi/__main__.py
?? .venv_tf/lib/python3.12/site-packages/certifi/cacert.pem
?? .venv_tf/lib/python3.12/site-packages/certifi/core.py
?? .venv_tf/lib/python3.12/site-packages/certifi/py.typed
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer-3.4.7.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/__init__.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/__main__.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/api.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/cd.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/cd.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/cli/__init__.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/cli/__main__.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/constant.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/legacy.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/md.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/md.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/models.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/py.typed
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/utils.py
?? .venv_tf/lib/python3.12/site-packages/charset_normalizer/version.py
?? .venv_tf/lib/python3.12/site-packages/clang/__init__.py
?? .venv_tf/lib/python3.12/site-packages/clang/cindex.py
?? .venv_tf/lib/python3.12/site-packages/clang/enumerations.py
?? .venv_tf/lib/python3.12/site-packages/clang/native/__init__.py
?? .venv_tf/lib/python3.12/site-packages/clang/native/libclang.so
?? .venv_tf/lib/python3.12/site-packages/dateutil/__init__.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/_common.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/_version.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/easter.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/parser/__init__.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/parser/_parser.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/parser/isoparser.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/relativedelta.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/rrule.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tz/__init__.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tz/_common.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tz/_factories.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tz/tz.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tz/win.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/tzwin.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/utils.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/zoneinfo/__init__.py
?? .venv_tf/lib/python3.12/site-packages/dateutil/zoneinfo/rebuild.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers-25.12.19.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/flatbuffers-25.12.19.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/flatbuffers-25.12.19.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/flatbuffers-25.12.19.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/flatbuffers-25.12.19.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/_version.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/builder.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/compat.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/encode.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/flexbuffers.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/number_types.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/packer.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/table.py
?? .venv_tf/lib/python3.12/site-packages/flatbuffers/util.py
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/gast-0.7.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/gast/__init__.py
?? .venv_tf/lib/python3.12/site-packages/gast/ast2.py
?? .venv_tf/lib/python3.12/site-packages/gast/ast3.py
?? .venv_tf/lib/python3.12/site-packages/gast/astn.py
?? .venv_tf/lib/python3.12/site-packages/gast/gast.py
?? .venv_tf/lib/python3.12/site-packages/gast/unparser.py
?? .venv_tf/lib/python3.12/site-packages/gast/version.py
?? .venv_tf/lib/python3.12/site-packages/google/_upb/_message.abi3.so
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/any.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/any_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/api_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/compiler/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/compiler/plugin_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/descriptor.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/descriptor_database.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/descriptor_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/descriptor_pool.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/duration.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/duration_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/empty_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/field_mask_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/api_implementation.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/builder.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/containers.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/decoder.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/encoder.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/enum_type_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/extension_dict.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/field_mask.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/message_listener.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/python_edition_defaults.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/python_message.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/testing_refleaks.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/type_checkers.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/well_known_types.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/internal/wire_format.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/json_format.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/message.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/message_factory.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/proto.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/proto_builder.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/proto_json.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/proto_text.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/pyext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/pyext/cpp_message.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/reflection.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/runtime_version.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/service_reflection.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/source_context_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/struct_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/symbol_database.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/testdata/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/text_encoding.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/text_format.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/timestamp.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/timestamp_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/type_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/unknown_fields.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/google/protobuf/wrappers_pb2.py
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/LICENSE
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/google_pasta-0.2.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/grpc/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_auth.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_channel.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_common.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_compression.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/_credentials/roots.pem
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/_cygrpc/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/_cygrpc/private_key_signing/private_key_signer_py_wrapper.cc
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/_cygrpc/private_key_signing/private_key_signer_py_wrapper.h
?? .venv_tf/lib/python3.12/site-packages/grpc/_cython/cygrpc.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/grpc/_grpcio_metadata.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_interceptor.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_observability.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_plugin_wrapping.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_runtime_protos.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_server.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_simple_stubs.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_typing.py
?? .venv_tf/lib/python3.12/site-packages/grpc/_utilities.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_base_call.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_base_channel.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_base_server.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_call.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_channel.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_interceptor.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_metadata.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_server.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_typing.py
?? .venv_tf/lib/python3.12/site-packages/grpc/aio/_utils.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/_client_adaptations.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/_metadata.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/_server_adaptations.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/implementations.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/interfaces.py
?? .venv_tf/lib/python3.12/site-packages/grpc/beta/utilities.py
?? .venv_tf/lib/python3.12/site-packages/grpc/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/experimental/aio/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/experimental/gevent.py
?? .venv_tf/lib/python3.12/site-packages/grpc/experimental/session_cache.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/common/cardinality.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/common/style.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/abandonment.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/callable_util.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/future.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/logging_pool.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/stream.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/foundation/stream_util.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/base/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/base/base.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/base/utilities.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/face/__init__.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/face/face.py
?? .venv_tf/lib/python3.12/site-packages/grpc/framework/interfaces/face/utilities.py
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/grpcio-1.80.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/licenses/hdf5.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/licenses/license.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/licenses/pytables.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/licenses/python.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/licenses/stdint.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/licenses/lzf/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/h5py-3.14.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/h5py.libs/libaec-7e9d22b8.so.0.1.3
?? .venv_tf/lib/python3.12/site-packages/h5py.libs/libhdf5-8c29085d.so.310.5.1
?? .venv_tf/lib/python3.12/site-packages/h5py.libs/libhdf5_hl-a45b8ce6.so.310.0.6
?? .venv_tf/lib/python3.12/site-packages/h5py.libs/libsz-b97e7bd9.so.2.0.1
?? .venv_tf/lib/python3.12/site-packages/h5py/__init__.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_conv.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/_errors.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/__init__.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/attrs.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/base.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/compat.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/dataset.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/datatype.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/dims.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/files.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/filters.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/group.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/selections.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/selections2.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_hl/vds.py
?? .venv_tf/lib/python3.12/site-packages/h5py/_npystrings.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/_objects.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/_proxy.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/_selector.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/defs.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5a.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5ac.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5d.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5ds.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5f.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5fd.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5g.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5i.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5l.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5o.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5p.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5pl.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5py_warnings.py
?? .venv_tf/lib/python3.12/site-packages/h5py/h5r.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5s.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5t.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/h5z.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/ipy_completer.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/common.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/conftest.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/data_files/__init__.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_attribute_create.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_attrs.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_attrs_data.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_base.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_big_endian_file.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_completions.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dataset.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dataset_getitem.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dataset_swmr.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_datatype.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dimension_scales.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dims_dimensionproxy.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_dtype.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_errors.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_file.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_file2.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_file_alignment.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_file_image.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_filters.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_group.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5d_direct_chunk.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5f.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5o.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5p.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5pl.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5s.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5t.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_h5z.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_npystrings.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_objects.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_ros3.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_selections.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_slicing.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_vds/__init__.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_vds/test_highlevel_vds.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_vds/test_lowlevel_vds.py
?? .venv_tf/lib/python3.12/site-packages/h5py/tests/test_vds/test_virtual_source.py
?? .venv_tf/lib/python3.12/site-packages/h5py/utils.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/h5py/version.py
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/idna-3.17.dist-info/licenses/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/idna/__init__.py
?? .venv_tf/lib/python3.12/site-packages/idna/__main__.py
?? .venv_tf/lib/python3.12/site-packages/idna/cli.py
?? .venv_tf/lib/python3.12/site-packages/idna/codec.py
?? .venv_tf/lib/python3.12/site-packages/idna/compat.py
?? .venv_tf/lib/python3.12/site-packages/idna/core.py
?? .venv_tf/lib/python3.12/site-packages/idna/idnadata.py
?? .venv_tf/lib/python3.12/site-packages/idna/intranges.py
?? .venv_tf/lib/python3.12/site-packages/idna/package_data.py
?? .venv_tf/lib/python3.12/site-packages/idna/py.typed
?? .venv_tf/lib/python3.12/site-packages/idna/uts46data.py
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/licenses/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/joblib-1.5.3.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/joblib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_cloudpickle_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_dask.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_memmapping_reducer.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_multiprocessing_helpers.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_parallel_backends.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_store_backends.py
?? .venv_tf/lib/python3.12/site-packages/joblib/_utils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/backports.py
?? .venv_tf/lib/python3.12/site-packages/joblib/compressor.py
?? .venv_tf/lib/python3.12/site-packages/joblib/disk.py
?? .venv_tf/lib/python3.12/site-packages/joblib/executor.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/cloudpickle/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/cloudpickle/cloudpickle.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/cloudpickle/cloudpickle_fast.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/_base.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/_posix_reduction.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/_win_reduction.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/context.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/fork_exec.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/popen_loky_posix.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/popen_loky_win32.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/process.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/queues.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/reduction.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/resource_tracker.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/spawn.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/synchronize.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/backend/utils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/cloudpickle_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/initializers.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/process_executor.py
?? .venv_tf/lib/python3.12/site-packages/joblib/externals/loky/reusable_executor.py
?? .venv_tf/lib/python3.12/site-packages/joblib/func_inspect.py
?? .venv_tf/lib/python3.12/site-packages/joblib/hashing.py
?? .venv_tf/lib/python3.12/site-packages/joblib/logger.py
?? .venv_tf/lib/python3.12/site-packages/joblib/memory.py
?? .venv_tf/lib/python3.12/site-packages/joblib/numpy_pickle.py
?? .venv_tf/lib/python3.12/site-packages/joblib/numpy_pickle_compat.py
?? .venv_tf/lib/python3.12/site-packages/joblib/numpy_pickle_utils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/parallel.py
?? .venv_tf/lib/python3.12/site-packages/joblib/pool.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/common.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/__init__.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/create_numpy_pickle.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_compressed_pickle_py27_np16.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_compressed_pickle_py27_np17.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_compressed_pickle_py33_np18.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_compressed_pickle_py34_np19.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_compressed_pickle_py35_np19.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl.bz2
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl.gzip
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl.lzma
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl.xz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl.bz2
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl.gzip
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl.lzma
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl.xz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl.bz2
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl.gzip
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl.lzma
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl.xz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl.bz2
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl.gzip
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl.lzma
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl.xz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_compressed_pickle_py36_np111.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl.bz2
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl.gzip
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl.lzma
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl.xz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.8.4_compressed_pickle_py27_np17.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_compressed_pickle_py27_np16.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_compressed_pickle_py27_np17.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_compressed_pickle_py34_np19.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_compressed_pickle_py35_np19.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_pickle_py27_np16.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_pickle_py27_np17.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_pickle_py33_np18.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_pickle_py34_np19.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.2_pickle_py35_np19.pkl
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_01.npy.z
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_02.npy.z
?? .venv_tf/lib/python3.12/site-packages/joblib/test/data/joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_03.npy.z
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_backports.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_cloudpickle_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_config.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_dask.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_disk.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_func_inspect.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_func_inspect_special_encoding.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_hashing.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_init.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_logger.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_memmapping.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_memory.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_memory_async.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_missing_multiprocessing.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_module.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_numpy_pickle.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_numpy_pickle_compat.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_numpy_pickle_utils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_parallel.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_store_backends.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_testing.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/test/testutils.py
?? .venv_tf/lib/python3.12/site-packages/joblib/testing.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/kagglehub-1.0.1.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/kagglehub/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/auth.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/cache.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/clients.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/colab_cache_resolver.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/competition.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/config.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/datasets.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/datasets_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/datasets_helpers.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/enum.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/env.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/exceptions.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/gcs_upload.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/handle.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/hf_datasets.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/http_resolver.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/integrity.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/kaggle_cache_resolver.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/logger.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/models.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/models_helpers.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/notebooks.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/packages.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/pandas_datasets.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/polars_datasets.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/registry.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/resolver.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/signing.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/tracker.py
?? .venv_tf/lib/python3.12/site-packages/kagglehub/utility_scripts.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk-0.1.27.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/kagglesdk-0.1.27.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/kagglesdk-0.1.27.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/kagglesdk-0.1.27.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/kagglesdk-0.1.27.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/abuse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/abuse/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/abuse/types/abuse_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/admin/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/admin/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/admin/services/inbox_file_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/admin/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/admin/types/inbox_file_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/services/agent_exam_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/types/agent_exam_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/agents/types/agent_exam_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/services/benchmark_tasks_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/services/benchmarks_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/benchmark_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/benchmark_task_run_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/benchmark_tasks_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/benchmark_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/benchmarks/types/benchmarks_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/blobs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/blobs/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/blobs/services/blob_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/blobs/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/blobs/types/blob_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/services/operations_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/cropped_image_upload.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/file_download.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/http_redirect.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/operations.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/common/types/operations_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/community/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/community/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/community/types/content_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/community/types/organization.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/services/competition_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/services/hackathon_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/services/hackathon_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/competition.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/competition_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/competition_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/episode.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/hackathon_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/hackathon_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/hackathons.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/page.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/page_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/search_competitions.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/submission_status.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/competitions/types/team.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/services/discussions_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/services/writeups_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/discussions_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/discussions_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/feedback_tracking_data.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/forum_message.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/search_discussions.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/writeup_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/writeup_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/discussions/types/writeups_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/services/education_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/types/education_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/education/types/education_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_client.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_creds.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_env.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_http_client.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_oauth.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kaggle_object.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/services/kernels_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/types/kernels_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/types/kernels_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/kernels/types/search_kernels.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/licenses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/licenses/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/licenses/types/licenses_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/services/model_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/services/model_proxy_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/services/model_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/model_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/model_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/model_proxy_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/model_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/model_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/models/types/search_models.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/services/search_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/types/search_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/types/search_content_shared.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/types/search_enums.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/search/types/search_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/services/iam_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/services/oauth_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/authentication.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/iam_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/oauth_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/roles.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/security/types/security_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/tags/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/tags/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/tags/types/tag_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/test/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/test/test_client.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/services/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/services/account_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/services/group_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/account_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/group_api_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/group_types.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/groups_enum.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/legacy_organizations_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/progression_service.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/search_users.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/user_avatar.py
?? .venv_tf/lib/python3.12/site-packages/kagglesdk/users/types/users_enums.py
?? .venv_tf/lib/python3.12/site-packages/keras-3.14.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/keras-3.14.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/keras-3.14.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/keras-3.14.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/keras-3.14.1.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/convnext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/densenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/efficientnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/efficientnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/imagenet_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/inception_resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/inception_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/mobilenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/mobilenet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/mobilenet_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/nasnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/resnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/resnet50/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/vgg16/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/vgg19/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/applications/xception/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/config/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/distillation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/distribution/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/dtype_policies/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/legacy/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/ops/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/ops/linalg/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/ops/nn/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/ops/numpy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/preprocessing/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/preprocessing/sequence/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/preprocessing/text/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/quantizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/random/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/tree/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/utils/bounding_boxes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/visualization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/_tf_keras/keras/wrappers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/convnext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/densenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/efficientnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/efficientnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/imagenet_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/inception_resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/inception_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/mobilenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/mobilenet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/mobilenet_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/nasnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/resnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/resnet50/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/vgg16/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/vgg19/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/applications/xception/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/config/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/distillation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/distribution/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/dtype_policies/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/legacy/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/ops/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/ops/linalg/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/ops/nn/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/ops/numpy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/preprocessing/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/preprocessing/sequence/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/quantizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/random/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/activations/activations.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/api_export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/convnext.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/densenet.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/efficientnet.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/efficientnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/imagenet_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/inception_resnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/inception_v3.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/mobilenet.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/mobilenet_v2.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/mobilenet_v3.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/nasnet.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/resnet.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/resnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/vgg16.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/vgg19.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/applications/xception.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/backend_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/dtypes.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/global_state.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/keras_tensor.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/masking.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/name_scope.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/remat.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/stateless_scope.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/symbolic_scope.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/tensor_attributes.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/common/variables.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/config.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/distribution_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/sparse.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/tensorboard.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/jax/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/numpy/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/openvino/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/distribution_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/sparse.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/tensorboard.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/trackable.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/tensorflow/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/export.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_adadelta.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_adagrad.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_adam.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_adamax.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_adamw.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_lion.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_nadam.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_parallel_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/optimizers/torch_sgd.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/backend/torch/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/backup_and_restore.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/callback.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/callback_list.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/csv_logger.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/early_stopping.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/history.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/lambda_callback.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/learning_rate_scheduler.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/model_checkpoint.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/monitor_callback.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/orbax_checkpoint.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/progbar_logger.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/reduce_lr_on_plateau.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/remote_monitor.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/swap_ema_weights.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/tensorboard.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/callbacks/terminate_on_nan.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/constraints/constraints.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/distillation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/distillation/distillation_loss.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/distillation/distiller.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/distribution/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/distribution/distribution_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/dtype_policies/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/dtype_policies/dtype_policy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/dtype_policies/dtype_policy_map.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/export_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/litert.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/neptune_model_export_archive.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/onnx.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/openvino.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/saved_model.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/saved_model_export_archive.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/tf2onnx_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/export/tfsm_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/initializers/constant_initializers.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/initializers/initializer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/initializers/random_initializers.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/activation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/elu.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/leaky_relu.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/prelu.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/relu.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/activations/softmax.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/attention/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/attention/additive_attention.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/attention/attention.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/attention/grouped_query_attention.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/attention/multi_head_attention.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv_transpose.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/base_depthwise_conv.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/base_separable_conv.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv1d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv2d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/conv3d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/depthwise_conv1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/depthwise_conv2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/separable_conv1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/convolutional/separable_conv2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/dense.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/einsum_dense.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/embedding.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/identity.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/input_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/lambda_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/masking.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/reversible_embedding.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/core/wrapper.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/input_spec.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/add.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/average.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/base_merge.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/concatenate.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/dot.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/maximum.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/minimum.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/multiply.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/merging/subtract.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/batch_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/group_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/layer_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/rms_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/spectral_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/normalization/unit_normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_average_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_average_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_average_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_max_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_max_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/adaptive_max_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/average_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/average_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/average_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/base_adaptive_pooling.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/base_global_pooling.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/base_pooling.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_average_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_average_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_average_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_max_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_max_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/global_max_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/max_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/max_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/pooling/max_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/category_encoding.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/data_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/discretization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/feature_space.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/hashed_crossing.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/hashing.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/aug_mix.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/auto_contrast.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/base_image_preprocessing_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/bounding_box.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/converters.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/formats.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/iou.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/bounding_boxes/validation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/center_crop.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/clahe.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/cut_mix.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/equalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/max_num_bounding_box.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/mix_up.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/rand_augment.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_brightness.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_color_degeneration.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_color_jitter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_contrast.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_crop.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_elastic_transform.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_erasing.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_flip.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_gaussian_blur.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_grayscale.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_hue.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_invert.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_perspective.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_posterization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_saturation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_sharpness.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_translation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_zoom.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/resizing.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/image_preprocessing/solarization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/index_lookup.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/integer_lookup.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/mel_spectrogram.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/normalization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/pipeline.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/rescaling.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/stft_spectrogram.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/string_lookup.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/preprocessing/text_vectorization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/activity_regularization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/alpha_dropout.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/dropout.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/gaussian_dropout.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/gaussian_noise.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/regularization/spatial_dropout.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/cropping1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/cropping2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/cropping3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/flatten.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/permute.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/repeat_vector.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/reshape.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/up_sampling1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/up_sampling2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/up_sampling3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/zero_padding1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/zero_padding2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/reshaping/zero_padding3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/bidirectional.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/conv_lstm.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/conv_lstm1d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/conv_lstm2d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/conv_lstm3d.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/dropout_rnn_cell.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/gru.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/lstm.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/simple_rnn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/stacked_rnn_cells.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/layers/rnn/time_distributed.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/backend.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/layers.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/losses.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/preprocessing/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/preprocessing/sequence.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/preprocessing/text.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/json_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/legacy_h5_format.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/saving_options.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/saving_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/legacy/saving/serialization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/losses/loss.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/losses/losses.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/accuracy_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/confusion_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/correlation_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/f_score_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/hinge_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/iou_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/metric.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/metrics_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/probabilistic_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/reduction_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/metrics/regression_metrics.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/cloning.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/functional.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/model.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/sequential.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/models/variable_mapping.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/einops.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/function.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/image.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/linalg.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/math.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/nn.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/node.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/numpy.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/operation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/operation_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/ops/symbolic_arguments.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adadelta.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adafactor.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adagrad.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adam.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adamax.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/adamw.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/base_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/ftrl.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/lamb.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/lion.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/loss_scale_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/muon.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/nadam.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/optimizer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/schedule_free_adamw.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/schedules/learning_rate_schedule.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/optimizers/sgd.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/awq.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/awq_config.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/awq_core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/gptq.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/gptq_config.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/gptq_core.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/quantization_config.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/quantizers.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/quantizers/utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/random/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/random/random.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/random/seed_generator.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/regularizers/regularizers.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/file_editor.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/keras_saveable.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/object_registration.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/orbax_util.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/saving_api.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/saving_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/saving/serialization_lib.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/testing/test_case.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/testing/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/compile_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/array_data_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/array_slicing.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/data_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/data_adapter_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/generator_data_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/grain_dataset_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/tf_dataset_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/data_adapters/torch_data_loader_adapter.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/trainers/trainer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/tree/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/tree/dmtree_impl.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/tree/optree_impl.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/tree/torchtree_impl.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/tree/tree_api.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/argument_validation.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/audio_dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/backend_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/code_stats.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/config.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/dtype_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/file_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/grain_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/image_dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/image_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/io_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/jax_layer.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/jax_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/model_visualization.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/module_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/naming.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/numerical_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/progbar.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/python_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/rng_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/sequence_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/summary_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/text_dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/tf_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/timeseries_dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/torch_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/traceback_utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/utils/tracking.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/version.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/draw_bounding_boxes.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/draw_segmentation_masks.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/plot_bounding_box_gallery.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/plot_image_gallery.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/visualization/plot_segmentation_mask_gallery.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/wrappers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/wrappers/fixes.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/wrappers/sklearn_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/keras/src/wrappers/utils.py
?? .venv_tf/lib/python3.12/site-packages/keras/tree/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/utils/bounding_boxes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/visualization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/keras/wrappers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/LICENSE.TXT
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/libclang-18.1.1.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/markdown_it/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/_compat.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/_punycode.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/cli/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/cli/parse.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/entities.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/html_blocks.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/html_re.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/normalize_url.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/common/utils.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/helpers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/helpers/parse_link_destination.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/helpers/parse_link_label.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/helpers/parse_link_title.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/main.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/parser_block.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/parser_core.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/parser_inline.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/port.yaml
?? .venv_tf/lib/python3.12/site-packages/markdown_it/presets/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/presets/commonmark.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/presets/default.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/presets/zero.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/py.typed
?? .venv_tf/lib/python3.12/site-packages/markdown_it/renderer.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/ruler.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/blockquote.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/code.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/fence.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/heading.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/hr.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/html_block.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/lheading.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/list.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/paragraph.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/reference.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/state_block.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_block/table.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/block.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/inline.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/linkify.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/normalize.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/replacements.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/smartquotes.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/state_core.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_core/text_join.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/__init__.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/autolink.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/backticks.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/balance_pairs.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/emphasis.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/entity.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/escape.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/fragments_join.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/html_inline.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/image.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/link.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/linkify.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/newline.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/state_inline.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/strikethrough.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/rules_inline/text.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/token.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/tree.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it/utils.py
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/markdown_it_py-4.2.0.dist-info/licenses/LICENSE.markdown-it
?? .venv_tf/lib/python3.12/site-packages/mdurl-0.1.2.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/mdurl-0.1.2.dist-info/LICENSE
?? .venv_tf/lib/python3.12/site-packages/mdurl-0.1.2.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/mdurl-0.1.2.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/mdurl-0.1.2.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/mdurl/__init__.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/_decode.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/_encode.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/_format.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/_parse.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/_url.py
?? .venv_tf/lib/python3.12/site-packages/mdurl/py.typed
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/licenses/LICENSE.eigen
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes-0.5.4.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes/_finfo.py
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes/_iinfo.py
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes/_ml_dtypes_ext.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/ml_dtypes/py.typed
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/namex-0.1.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/namex/__init__.py
?? .venv_tf/lib/python3.12/site-packages/namex/convert.py
?? .venv_tf/lib/python3.12/site-packages/namex/export.py
?? .venv_tf/lib/python3.12/site-packages/namex/generate.py
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/include/numpy/libdivide/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/src/common/pythoncapi-compat/COPYING
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/src/highway/LICENSE
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/src/multiarray/dragon4_LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/src/npysort/x86-simd-sort/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/_core/src/umath/svml/LICENSE
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/fft/pocketfft/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/linalg/lapack_lite/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/ma/LICENSE
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/distributions/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/mt19937/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/pcg64/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/philox/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/sfc64/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy-2.4.6.dist-info/licenses/numpy/random/src/splitmix64/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0
?? .venv_tf/lib/python3.12/site-packages/numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0
?? .venv_tf/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-32a4b2a6.so
?? .venv_tf/lib/python3.12/site-packages/numpy/__config__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/__config__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/__init__.cython-30.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/__init__.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_array_api_info.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_array_api_info.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_configtool.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_configtool.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_add_newdocs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_add_newdocs.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_add_newdocs_scalars.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_add_newdocs_scalars.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_asarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_asarray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_dtype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_dtype.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_dtype_ctypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_dtype_ctypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_exceptions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_exceptions.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_internal.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_internal.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_methods.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_methods.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_multiarray_tests.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_operand_flag_tests.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_rational_tests.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_simd.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_simd.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_string_helpers.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_string_helpers.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_struct_ufunc_tests.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_type_aliases.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_type_aliases.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_ufunc_config.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_ufunc_config.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_umath_tests.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/_umath_tests.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/arrayprint.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/arrayprint.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/cversions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/defchararray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/defchararray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/einsumfunc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/einsumfunc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/fromnumeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/fromnumeric.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/function_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/function_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/getlimits.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/getlimits.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/__multiarray_api.c
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/__multiarray_api.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/__ufunc_api.c
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/__ufunc_api.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/_neighborhood_iterator_imp.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/_numpyconfig.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/_public_dtype_api_table.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/arrayobject.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/arrayscalars.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/dtype_api.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/halffloat.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/ndarrayobject.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/ndarraytypes.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_2_compat.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_2_complexcompat.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_3kcompat.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_common.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_cpu.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_endian.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_math.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_no_deprecated_api.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/npy_os.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/numpyconfig.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/random/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/random/bitgen.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/random/distributions.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/random/libdivide.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/ufuncobject.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/include/numpy/utils.h
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/lib/libnpymath.a
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/lib/npy-pkg-config/mlib.ini
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/lib/npy-pkg-config/npymath.ini
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/lib/pkgconfig/numpy.pc
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/memmap.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/memmap.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/multiarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/multiarray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/numeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/numeric.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/numerictypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/numerictypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/overrides.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/overrides.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/printoptions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/printoptions.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/records.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/records.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/shape_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/shape_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/strings.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/strings.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/_locales.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/_natype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/astype_copy.pkl
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/generate_umath_validation_data.cpp
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/recarray_from_file.fits
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-README.txt
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arccos.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arccosh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arcsin.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arcsinh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arctan.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-arctanh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-cbrt.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-cos.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-cosh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-exp.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-exp2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-expm1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-log.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-log10.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-log1p.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-log2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-sin.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-sinh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-tan.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/data/umath-validation-set-tanh.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/cython/checks.pyx
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/cython/meson.build
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/cython/setup.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/limited_api/limited_api1.c
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/limited_api/limited_api2.pyx
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/limited_api/limited_api_latest.c
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/limited_api/meson.build
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/examples/limited_api/setup.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test__exceptions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_abc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_api.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_argparse.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_array_api_info.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_array_coercion.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_array_interface.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_arraymethod.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_arrayobject.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_arrayprint.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_casting_floatingpoint_errors.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_casting_unittests.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_conversion_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_cpu_dispatcher.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_cpu_features.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_custom_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_cython.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_defchararray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_deprecations.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_dlpack.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_dtype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_einsum.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_errstate.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_extint128.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_finfo.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_function_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_getlimits.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_half.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_hashtable.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_indexerrors.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_item_selection.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_limited_api.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_longdouble.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_mem_overlap.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_mem_policy.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_memmap.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_multiprocessing.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_multithreading.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_nditer.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_nep50_promotions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_numeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_numerictypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_overrides.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_print.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_protocols.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_records.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalar_ctors.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalar_methods.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalarbuffer.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalarinherit.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalarmath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_scalarprint.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_shape_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_simd_module.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_stringdtype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_strings.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_umath_accuracy.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/tests/test_unicode.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/umath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_core/umath.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_distributor_init.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_distributor_init.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_expired_attrs_2_0.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_expired_attrs_2_0.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_globals.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_globals.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/hook-numpy.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/hook-numpy.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/tests/pyinstaller-smoke.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pyinstaller/tests/test_pyinstaller.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pytesttester.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_pytesttester.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_add_docstring.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_array_like.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_char_codes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_dtype_like.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_extended_precision.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_nbit.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_nbit_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_nbit_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_nested_sequence.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_scalars.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_shape.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_ufunc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_typing/_ufunc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_convertions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_convertions.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_inspect.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_inspect.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_pep440.py
?? .venv_tf/lib/python3.12/site-packages/numpy/_utils/_pep440.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/char/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/char/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/conftest.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_dtype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_dtype.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_dtype_ctypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_dtype_ctypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_internal.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_multiarray_umath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/arrayprint.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/defchararray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/einsumfunc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/fromnumeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/function_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/getlimits.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/multiarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/numeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/numerictypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/overrides.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/overrides.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/core/records.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/shape_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/core/umath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ctypeslib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ctypeslib/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/ctypeslib/_ctypeslib.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ctypeslib/_ctypeslib.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/doc/ufuncs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/dtypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/dtypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/exceptions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/exceptions.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/__main__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/__version__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/__version__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_backend.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_backend.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_distutils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_distutils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_meson.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/_meson.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_backends/meson.build.template
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_isocbind.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_isocbind.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_src_pyf.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/_src_pyf.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/auxfuncs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/auxfuncs.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/capi_maps.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/capi_maps.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/cb_rules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/cb_rules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/cfuncs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/cfuncs.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/common_rules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/common_rules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/crackfortran.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/crackfortran.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/diagnose.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/diagnose.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/f2py2e.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/f2py2e.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/f90mod_rules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/f90mod_rules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/func2subr.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/func2subr.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/rules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/rules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/setup.cfg
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/src/fortranobject.c
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/src/fortranobject.h
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/symbolic.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/symbolic.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/abstract_interface/foo.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/abstract_interface/gh18403_mod.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/array_from_pyobj/wrapmodule.c
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/assumed_shape/.f2py_f2cmap
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/assumed_shape/foo_free.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/assumed_shape/foo_mod.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/assumed_shape/foo_use.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/assumed_shape/precision.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/block_docstring/foo.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/foo.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/gh17797.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/gh18335.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/gh25211.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/gh25211.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/callback/gh26681.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/cli/gh_22819.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/cli/hi77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/cli/hiworld.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/common/block.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/common/gh19161.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/accesstype.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/common_with_division.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/data_common.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/data_multiplier.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/data_stmts.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/data_with_comments.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/foo_deps.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh15035.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh17859.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh22648.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh23533.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh23598.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh23598Warn.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh23879.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh27697.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/gh2848.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/operators.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/privatemod.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/publicmod.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/pubprivmod.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/crackfortran/unicode_comment.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/f2cmap/.f2py_f2cmap
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/f2cmap/isoFortranEnvMap.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/isocintrin/isoCtests.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/kind/foo.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/mixed/foo.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/mixed/foo_fixed.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/mixed/foo_free.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/gh25337/data.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/gh25337/use_data.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/gh26920/two_mods_with_no_public_entities.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/gh26920/two_mods_with_one_public_routine.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/module_data_docstring.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/modules/use_modules.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/negative_bounds/issue_20853.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_array.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_both.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_compound.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_integer.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_non_compound.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/parameter/constant_real.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/quoted_character/foo.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/AB.inc
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/assignOnlyModule.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/datonly.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/f77comments.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/f77fixedform.f95
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/f90continuation.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/incfile.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/inout.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/lower_f2py_fortran.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/regression/mod_derived_types.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_character/foo77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_character/foo90.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_complex/foo77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_complex/foo90.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_integer/foo77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_integer/foo90.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_logical/foo77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_logical/foo90.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_real/foo77.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/return_real/foo90.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/routines/funcfortranname.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/routines/funcfortranname.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/routines/subrout.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/routines/subrout.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/size/foo.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/char.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/fixed_string.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/gh24008.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/gh24662.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/gh25286.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/gh25286.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/gh25286_bc.pyf
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/scalar_string.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/string/string.f
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/src/value_attrspec/gh21665.f90
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_abstract_interface.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_array_from_pyobj.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_assumed_shape.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_block_docstring.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_callback.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_capi_maps.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_character.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_common.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_crackfortran.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_data.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_docs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_f2cmap.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_isoc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_kind.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_mixed.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_modules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_parameter.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_pyf_src.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_quoted_character.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_return_character.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_return_complex.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_return_integer.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_return_logical.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_return_real.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_routines.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_semicolon_split.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_size.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_string.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_symbolic.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/test_value_attrspec.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/tests/util.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/use_rules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/f2py/use_rules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/_helper.py
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/_helper.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/_pocketfft.py
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/_pocketfft.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/_pocketfft_umath.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/tests/test_helper.py
?? .venv_tf/lib/python3.12/site-packages/numpy/fft/tests/test_pocketfft.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_array_utils_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_array_utils_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arraypad_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arraypad_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arraysetops_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arraysetops_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arrayterator_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_arrayterator_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_datasource.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_datasource.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_format_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_format_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_function_base_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_histograms_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_histograms_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_index_tricks_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_index_tricks_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_iotools.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_iotools.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_npyio_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_scimath_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_scimath_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_shape_base_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_shape_base_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_twodim_base_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_twodim_base_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_type_check_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_type_check_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_ufunclike_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_ufunclike_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_user_array_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_user_array_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_utils_impl.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_utils_impl.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_version.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/_version.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/array_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/array_utils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/format.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/format.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/introspect.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/introspect.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/mixins.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/mixins.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/npyio.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/npyio.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/recfunctions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/recfunctions.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/scimath.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/scimath.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/stride_tricks.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/stride_tricks.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test__datasource.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test__iotools.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test__version.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_array_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_arraypad.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_arraysetops.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_arrayterator.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_format.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_function_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_histograms.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_index_tricks.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_io.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_loadtxt.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_mixins.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_packbits.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_polynomial.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_recfunctions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_shape_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_stride_tricks.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_twodim_base.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_type_check.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_ufunclike.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/tests/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/user_array.py
?? .venv_tf/lib/python3.12/site-packages/numpy/lib/user_array.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/_linalg.py
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/_linalg.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/lapack_lite.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/lapack_lite.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/tests/test_deprecations.py
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py
?? .venv_tf/lib/python3.12/site-packages/numpy/linalg/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/API_CHANGES.txt
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/LICENSE
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/README.rst
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/core.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/core.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/extras.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/extras.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/mrecords.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/mrecords.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_arrayobject.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_core.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_deprecations.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_extras.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_mrecords.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_old_ma.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/tests/test_subclassing.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/testutils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/ma/testutils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/matlib.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matlib.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/defmatrix.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/defmatrix.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_defmatrix.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_interaction.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_masked_matrix.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_matrix_linalg.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_multiarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_numeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/matrixlib/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/_polybase.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/_polybase.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/_polytypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/chebyshev.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/chebyshev.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/hermite.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/hermite.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/hermite_e.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/hermite_e.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/laguerre.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/laguerre.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/legendre.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/legendre.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/polynomial.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/polynomial.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/polyutils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/polyutils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_classes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_polyutils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_printing.py
?? .venv_tf/lib/python3.12/site-packages/numpy/polynomial/tests/test_symbol.py
?? .venv_tf/lib/python3.12/site-packages/numpy/py.typed
?? .venv_tf/lib/python3.12/site-packages/numpy/random/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/numpy/random/__init__.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/random/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_bounded_integers.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_bounded_integers.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_common.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_common.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_common.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/cffi/extending.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/cffi/parse.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/cython/extending.pyx
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/cython/extending_distributions.pyx
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/cython/meson.build
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/numba/extending.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_examples/numba/extending_distributions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_generator.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_mt19937.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_pcg64.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_philox.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_pickle.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_pickle.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/_sfc64.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/bit_generator.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/random/bit_generator.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/c_distributions.pxd
?? .venv_tf/lib/python3.12/site-packages/numpy/random/lib/libnpyrandom.a
?? .venv_tf/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/numpy/random/mtrand.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/generator_pcg64_np121.pkl.gz
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/generator_pcg64_np126.pkl.gz
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/mt19937-testset-1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/mt19937-testset-2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/pcg64-testset-1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/pcg64-testset-2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/pcg64dxsm-testset-1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/pcg64dxsm-testset-2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/philox-testset-1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/philox-testset-2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/sfc64-testset-1.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/sfc64-testset-2.csv
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/data/sfc64_np126.pkl.gz
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_direct.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_extending.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937_regressions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_random.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_randomstate.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_randomstate_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_regression.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_seed_sequence.py
?? .venv_tf/lib/python3.12/site-packages/numpy/random/tests/test_smoke.py
?? .venv_tf/lib/python3.12/site-packages/numpy/rec/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/rec/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/strings/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/strings/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/extbuild.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/extbuild.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/_private/utils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/overrides.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/overrides.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/print_coercion_tables.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/print_coercion_tables.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/testing/tests/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test__all__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_configtool.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_ctypeslib.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_lazyloading.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_matlib.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_numpy_config.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_numpy_version.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_public_api.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_reloading.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_scripts.py
?? .venv_tf/lib/python3.12/site-packages/numpy/tests/test_warnings.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/mypy_plugin.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/arithmetic.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/array_constructors.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/array_like.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/array_pad.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/arrayprint.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/arrayterator.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/bitwise_ops.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/char.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/chararray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/comparisons.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/constants.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/datasource.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/dtype.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/einsumfunc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/flatiter.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/fromnumeric.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/histograms.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/index_tricks.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/lib_function_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/lib_polynomial.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/lib_utils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/lib_version.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/linalg.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ma.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/memmap.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/modules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/multiarray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ndarray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ndarray_misc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/nditer.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/nested_sequence.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/npyio.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/numerictypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/random.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/rec.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/scalars.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/shape.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/shape_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/stride_tricks.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/strings.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/testing.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/twodim_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/type_check.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ufunc_config.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ufunclike.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/ufuncs.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/fail/warnings_and_errors.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/misc/extended_precision.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/mypy.ini
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/array_constructors.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/array_like.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/arrayprint.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/arrayterator.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/bitwise_ops.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/comparisons.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/dtype.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/einsumfunc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/flatiter.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/fromnumeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/index_tricks.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/lib_user_array.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/lib_utils.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/lib_version.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/literal.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ma.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/mod.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/modules.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/multiarray.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ndarray_conversion.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ndarray_misc.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ndarray_shape_manipulation.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/nditer.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/numeric.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/numerictypes.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/random.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/recfunctions.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/scalars.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/shape.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/simple.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ufunc_config.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ufunclike.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/ufuncs.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/pass/warnings_and_errors.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/arithmetic.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/array_api_info.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/array_constructors.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/arraypad.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/arrayprint.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/arraysetops.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/arrayterator.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/bitwise_ops.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/char.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/chararray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/comparisons.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/constants.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ctypeslib.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/datasource.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/dtype.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/einsumfunc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/emath.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/fft.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/flatiter.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/fromnumeric.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/getlimits.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/histograms.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/index_tricks.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/lib_function_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/lib_polynomial.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/lib_utils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/lib_version.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/linalg.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ma.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/matrix.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/memmap.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/mod.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/modules.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/multiarray.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/nbit_base_example.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ndarray_assignability.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ndarray_conversion.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ndarray_misc.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ndarray_shape_manipulation.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/nditer.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/nested_sequence.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/npyio.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/numeric.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/numerictypes.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/polynomial_polybase.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/polynomial_polyutils.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/polynomial_series.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/random.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/rec.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/scalars.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/shape.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/shape_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/stride_tricks.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/strings.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/testing.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/twodim_base.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/type_check.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ufunc_config.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ufunclike.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/ufuncs.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/data/reveal/warnings_and_errors.pyi
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/test_isfile.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/test_runtime.py
?? .venv_tf/lib/python3.12/site-packages/numpy/typing/tests/test_typing.py
?? .venv_tf/lib/python3.12/site-packages/numpy/version.py
?? .venv_tf/lib/python3.12/site-packages/numpy/version.pyi
?? .venv_tf/lib/python3.12/site-packages/nvidia/__init__.py
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/cublas.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/cublasLt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/cublasXt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/cublas_api.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/cublas_v2.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/include/nvblas.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cublas/lib/libnvblas.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/Openacc/cupti_openacc.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/Openmp/cupti_openmp.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/Openmp/omp-tools.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cuda_stdint.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_activity.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_activity_deprecated.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_callbacks.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_checkpoint.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_driver_cbid.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_events.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_metrics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_nvtx_cbid.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_pcsampling.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_pcsampling_util.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_pmsampling.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_profiler_host.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_profiler_target.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_range_profiler.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_result.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_runtime_cbid.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_sass_metrics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_target.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/cupti_version.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cudaGL_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cudaVDPAU_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cuda_gl_interop_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cuda_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cuda_runtime_api_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cuda_vdpau_interop_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_cudart_removed_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/generated_nvtx_meta.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/nvperf_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/nvperf_cuda_host.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/nvperf_host.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/include/nvperf_target.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcheckpoint.so
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libnvperf_host.so
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libnvperf_target.so
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libpcsamplingutil.so
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/common_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/cudacc_ext.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/device_double_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/device_double_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/device_fp128_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/device_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/device_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/func_macro.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/host_config.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/host_defines.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/host_runtime.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/math_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/math_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/mma.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/mma.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_100_rt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_100_rt.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_70_rt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_70_rt.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_80_rt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_80_rt.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_90_rt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/sm_90_rt.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/crt/storage_class.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/fatbinary_section.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/include/nvPTXCompiler.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/include/nvvm.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/include/nvrtc.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.alt.so.12.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.alt.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/builtin_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/channel_descriptor.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/common_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/async.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/coalesced_reduce.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/coalesced_scan.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/driver_abi.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/functional.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/helpers.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/info.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/invoke.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/memory.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/partitioning.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/reduce.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/scan.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/details/sync.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/memcpy_async.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/reduce.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cooperative_groups/scan.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuComplex.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaEGL.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaEGLTypedefs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaGL.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaGLTypedefs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaProfilerTypedefs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaTypedefs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaVDPAU.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudaVDPAUTypedefs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_awbarrier.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_awbarrier_helpers.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_awbarrier_primitives.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_bf16.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_bf16.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_device_runtime_api.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_egl_interop.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp16.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp4.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp4.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp6.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp6.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp8.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_fp8.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_gl_interop.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_occupancy.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_pipeline.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_pipeline_helpers.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_pipeline_primitives.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_runtime_api.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_surface_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_texture_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cuda_vdpau_interop.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/cudart_platform.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_atomic_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_atomic_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_double_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_launch_parameters.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/device_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/driver_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/driver_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/host_config.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/host_defines.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/library_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/math_constants.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/math_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/mma.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_20_atomic_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_20_atomic_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_20_intrinsics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_20_intrinsics.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_30_intrinsics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_30_intrinsics.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_32_atomic_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_32_atomic_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_32_intrinsics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_32_intrinsics.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_35_atomic_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_35_intrinsics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_60_atomic_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_60_atomic_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_61_intrinsics.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/sm_61_intrinsics.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/surface_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/surface_indirect_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/surface_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/texture_fetch_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/texture_indirect_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/texture_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/vector_functions.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/vector_functions.hpp
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/include/vector_types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudadevrt.a
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart_static.a
?? .venv_tf/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libculibos.a
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_adv.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_adv_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_backend.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_backend_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_cnn.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_cnn_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_graph.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_graph_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_ops.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_ops_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_subquadratic_ops.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_subquadratic_ops_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_version.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn_version_v9.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_engines_tensor_ir.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_ext.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_graph.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_heuristic.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/include/cudalibxt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/include/cufft.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/include/cufftXt.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/include/cufftw.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/lib/libcufft.so.11
?? .venv_tf/lib/python3.12/site-packages/nvidia/cufft/lib/libcufftw.so.11
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/__init__.py
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/__init__.py
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_discrete.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_discrete2.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_globals.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_kernel.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_lognormal.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_mrg32k3a.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_mtgp32.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_mtgp32_host.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_mtgp32_kernel.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_mtgp32dc_p_11213.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_normal.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_normal_static.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_philox4x32_x.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_poisson.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_precalc.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/include/curand_uniform.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/lib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/nvidia/curand/lib/libcurand.so.10
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolverDn.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolverMg.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolverRf.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolverSp.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolverSp_LOWLEVEL_PREVIEW.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/include/cusolver_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/lib/libcusolver.so.11
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusolver/lib/libcusolverMg.so.11
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusparse/include/cusparse.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusparse/include/cusparse_v2.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/cusparse/lib/libcusparse.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/barrier.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/comm.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/coop.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/core.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/common/doca_gpunetio_verbs_def.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/common/doca_gpunetio_verbs_dev.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_common.cuh
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_counter.cuh
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_cq.cuh
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_onesided.cuh
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_qp.cuh
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/doca_gpunetio/doca_gpunetio_device.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/gin_gdaki.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gdaki/gin_gdaki_device_host_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gin_device_api.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gin_device_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/gin_device_host_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/proxy/gin_proxy.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin_barrier.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/gin_win_stub.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/barrier__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/barrier__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/comm__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/comm__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/core__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/core__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/gin__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/gin__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/gin_barrier__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/gin_barrier__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/ll_a2a__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/ll_a2a__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/lsa_barrier__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/lsa_barrier__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/multimem__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/ptr__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/ptr__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/reduce_copy__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/reduce_copy__impl.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/reduce_copy__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/vector__funcs.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/impl/vector__types.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/ll_a2a.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/lsa_barrier.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/net_device.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/ptr.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/reduce_copy.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device/utility.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/include/nccl_device_wrapper.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
?? .venv_tf/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl_device.bc
?? .venv_tf/lib/python3.12/site-packages/nvidia/nvjitlink/include/nvJitLink.h
?? .venv_tf/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cublas_cu12-12.9.2.10.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_cupti_cu12-12.9.79.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvcc_cu12-12.9.86.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cuda_runtime_cu12-12.9.79.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cudnn_cu12-9.23.0.39.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cufft_cu12-11.4.1.4.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_curand_cu12-10.3.10.19.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusolver_cu12-11.7.5.82.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_cusparse_cu12-12.5.10.65.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_nccl_cu12-2.30.4.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/licenses/License.txt
?? .venv_tf/lib/python3.12/site-packages/nvidia_nvjitlink_cu12-12.9.86.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/opt_einsum-3.4.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/opt_einsum-3.4.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/opt_einsum-3.4.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/opt_einsum-3.4.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/opt_einsum-3.4.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/__init__.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/_version.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/__init__.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/cupy.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/dispatch.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/jax.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/object_arrays.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/tensorflow.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/theano.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/backends/torch.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/blas.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/contract.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/helpers.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/parser.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/path_random.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/paths.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/sharing.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/testing.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_backends.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_blas.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_contract.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_edge_cases.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_input.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_parser.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_paths.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/tests/test_sharing.py
?? .venv_tf/lib/python3.12/site-packages/opt_einsum/typing.py
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/optree-0.19.1.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/optree/_C.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/optree/_C.pyi
?? .venv_tf/lib/python3.12/site-packages/optree/__init__.py
?? .venv_tf/lib/python3.12/site-packages/optree/accessors.py
?? .venv_tf/lib/python3.12/site-packages/optree/dataclasses.py
?? .venv_tf/lib/python3.12/site-packages/optree/functools.py
?? .venv_tf/lib/python3.12/site-packages/optree/integrations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/optree/integrations/attrs.py
?? .venv_tf/lib/python3.12/site-packages/optree/integrations/jax.py
?? .venv_tf/lib/python3.12/site-packages/optree/integrations/numpy.py
?? .venv_tf/lib/python3.12/site-packages/optree/integrations/torch.py
?? .venv_tf/lib/python3.12/site-packages/optree/ops.py
?? .venv_tf/lib/python3.12/site-packages/optree/py.typed
?? .venv_tf/lib/python3.12/site-packages/optree/pytree.py
?? .venv_tf/lib/python3.12/site-packages/optree/registry.py
?? .venv_tf/lib/python3.12/site-packages/optree/treespec.py
?? .venv_tf/lib/python3.12/site-packages/optree/typing.py
?? .venv_tf/lib/python3.12/site-packages/optree/utils.py
?? .venv_tf/lib/python3.12/site-packages/optree/version.py
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/licenses/LICENSE.APACHE
?? .venv_tf/lib/python3.12/site-packages/packaging-26.2.dist-info/licenses/LICENSE.BSD
?? .venv_tf/lib/python3.12/site-packages/packaging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_elffile.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_manylinux.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_musllinux.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_parser.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_structures.py
?? .venv_tf/lib/python3.12/site-packages/packaging/_tokenizer.py
?? .venv_tf/lib/python3.12/site-packages/packaging/dependency_groups.py
?? .venv_tf/lib/python3.12/site-packages/packaging/direct_url.py
?? .venv_tf/lib/python3.12/site-packages/packaging/errors.py
?? .venv_tf/lib/python3.12/site-packages/packaging/licenses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/packaging/licenses/_spdx.py
?? .venv_tf/lib/python3.12/site-packages/packaging/markers.py
?? .venv_tf/lib/python3.12/site-packages/packaging/metadata.py
?? .venv_tf/lib/python3.12/site-packages/packaging/py.typed
?? .venv_tf/lib/python3.12/site-packages/packaging/pylock.py
?? .venv_tf/lib/python3.12/site-packages/packaging/requirements.py
?? .venv_tf/lib/python3.12/site-packages/packaging/specifiers.py
?? .venv_tf/lib/python3.12/site-packages/packaging/tags.py
?? .venv_tf/lib/python3.12/site-packages/packaging/utils.py
?? .venv_tf/lib/python3.12/site-packages/packaging/version.py
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/pandas-3.0.3.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/pandas/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_config/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_config/config.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_config/dates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_config/display.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_config/localization.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/_cyutility.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/algos.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/algos.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/arrays.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/arrays.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/byteswap.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/byteswap.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/groupby.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/groupby.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/hashing.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/hashing.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/hashtable.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/hashtable.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/index.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/index.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/indexing.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/indexing.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/internals.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/internals.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/interval.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/interval.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/join.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/join.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/json.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/json.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/lib.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/lib.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/missing.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/missing.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/ops.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/ops.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/ops_dispatch.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/ops_dispatch.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/pandas_datetime.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/pandas_parser.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/parsers.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/parsers.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/properties.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/properties.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/reshape.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/reshape.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/sas.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/sas.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/sparse.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/sparse.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/testing.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/testing.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslib.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslib.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/base.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/ccalendar.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/ccalendar.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/conversion.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/conversion.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/dtypes.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/dtypes.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/fields.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/fields.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/nattype.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/nattype.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/np_datetime.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/np_datetime.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/offsets.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/offsets.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/parsing.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/parsing.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/period.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/period.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/strptime.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/strptime.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timedeltas.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timedeltas.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timestamps.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timestamps.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timezones.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/timezones.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/tzconversion.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/tzconversion.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/vectorized.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/tslibs/vectorized.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/window/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/window/aggregations.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/window/aggregations.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/window/indexers.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/window/indexers.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/writers.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/pandas/_libs/writers.pyi
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/_hypothesis.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/_io.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/_warnings.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/asserters.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_testing/contexts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_typing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_version.py
?? .venv_tf/lib/python3.12/site-packages/pandas/_version_meson.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/executors/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/extensions/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/indexers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/interchange/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/internals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/typing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/api/typing/aliases.py
?? .venv_tf/lib/python3.12/site-packages/pandas/arrays/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/_constants.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/_optional.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/numpy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/numpy/function.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/pickle_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/compat/pyarrow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/executor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/extensions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/mean_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/min_max_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/shared.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/sum_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/_numba/kernels/var_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/algorithms.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/apply.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/datetimelike_accumulations.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/masked_accumulations.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/masked_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/putmask.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/quantile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/array_algos/transforms.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arraylike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/_arrow_string_mixins.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/_mixins.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/_ranges.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/_utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/arrow/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/arrow/_arrow_utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/arrow/accessors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/arrow/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/arrow/extension_types.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/boolean.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/datetimes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/floating.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/integer.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/masked.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/numeric.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/numpy_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/sparse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/sparse/accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/sparse/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/sparse/scipy_sparse.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/string_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/string_arrow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/arrays/timedeltas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/col.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/align.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/check.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/engines.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/eval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/expr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/expressions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/parsing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/pytables.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/computation/scope.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/config_init.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/construction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/cast.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/generic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/inference.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/dtypes/missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/flags.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/frame.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/generic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/generic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/grouper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/numba_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/groupby/ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexers/objects.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexers/utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/accessors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/category.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/datetimes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/extension.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/frozen.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/multi.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexes/timedeltas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/buffer.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/column.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/dataframe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/dataframe_protocol.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/from_dataframe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/interchange/utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/blocks.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/construction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/managers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/internals/ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/methods/describe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/methods/selectn.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/methods/to_dict.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/nanops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/array_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/dispatch.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/docstrings.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/invalid.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/mask_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/ops/missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/resample.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/encoding.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/melt.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/merge.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/pivot.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/reshape.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/reshape/tile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/roperator.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/sample.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/shared_docs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/sorting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/sparse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/sparse/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/strings/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/strings/accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/strings/object_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/tools/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/tools/datetimes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/tools/numeric.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/tools/timedeltas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/tools/times.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/util/hashing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/util/numba_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/doc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/ewm.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/expanding.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/numba_.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/online.py
?? .venv_tf/lib/python3.12/site-packages/pandas/core/window/rolling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/errors/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/errors/cow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/clipboard/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/clipboards.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_calamine.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_odfreader.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_odswriter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_openpyxl.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_pyxlsb.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_xlrd.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/excel/_xlsxwriter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/feather_format.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/_color_data.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/console.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/css.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/csvs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/excel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/format.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/html.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/info.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/printing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/style.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/style_render.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/html.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/html_style.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/html_table.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/latex.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/latex_longtable.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/latex_table.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/string.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/templates/typst.tpl
?? .venv_tf/lib/python3.12/site-packages/pandas/io/formats/xml.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/html.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/iceberg.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/json/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/json/_json.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/json/_normalize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/json/_table_schema.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/orc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parquet.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/arrow_parser_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/base_parser.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/c_parser_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/python_parser.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/parsers/readers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/pytables.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sas/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sas/sas7bdat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sas/sas_constants.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sas/sas_xport.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sas/sasreader.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/spss.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/sql.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/stata.py
?? .venv_tf/lib/python3.12/site-packages/pandas/io/xml.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_core.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/boxplot.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/converter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/hist.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/misc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/style.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/timeseries.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_matplotlib/tools.py
?? .venv_tf/lib/python3.12/site-packages/pandas/plotting/_misc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/pyproject.toml
?? .venv_tf/lib/python3.12/site-packages/pandas/testing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/api/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/api/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/api/test_types.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_frame_apply.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_frame_apply_relabeling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_frame_transform.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_invalid_arg.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_series_apply.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_series_apply_relabeling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_series_transform.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/apply/test_str.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_bool.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_datetime64.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_numeric.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_object.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arithmetic/test_timedelta64.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_comparison.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_construction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_function.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_logical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_reduction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/boolean/test_repr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_algos.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_analytics.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_map.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_operators.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_repr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_sorting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/categorical/test_warnings.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/datetimes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/datetimes/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/datetimes/test_cumulative.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/datetimes/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_comparison.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_construction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_contains.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_function.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_repr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/floating/test_to_numpy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_comparison.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_construction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_function.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_reduction.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/integer/test_repr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/test_interval_pyarrow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/interval/test_overlaps.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked/test_arrow_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked/test_function.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/masked_shared.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/numpy_/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/numpy_/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/numpy_/test_numpy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/period/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/period/test_arrow_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/period/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/period/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/period/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_arithmetics.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_combine_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_dtype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_libsparse.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/sparse/test_unary.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/string_/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/string_/test_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/string_/test_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/string_/test_string_arrow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_datetimes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_ndarray_backed.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/test_timedeltas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/timedeltas/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/timedeltas/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/timedeltas/test_cumulative.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/arrays/timedeltas/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_conversion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_misc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_transpose.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_unique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/base/test_value_counts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/computation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/computation/test_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/computation/test_eval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/config/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/config/test_config.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/config/test_localization.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/construction/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/construction/test_extract_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/test_datetimeindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/test_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/test_intervalindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/test_periodindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/index/test_timedeltaindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_chained_assignment_deprecation.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_clip.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_copy_deprecation.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_core_functionalities.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_functions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_internals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_interp_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_methods.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_setitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/test_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/copy_view/util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_box_unbox.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_can_hold_element.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_construct_from_scalar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_construct_ndarray.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_construct_object_arr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_dict_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_downcast.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_find_common_type.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_infer_datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_infer_dtype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/cast/test_promote.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_generic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_inference.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/dtypes/test_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/array_with_attr/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/array_with_attr/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/array_with_attr/test_array_with_attr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/accumulate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/casting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/dim2.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/dtype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/getitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/interface.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/io.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/methods.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/printing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/reduce.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/reshaping.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/base/setitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/date/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/date/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/decimal/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/decimal/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/decimal/test_decimal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/json/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/json/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/json/test_json.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/list/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/list/array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/list/test_list.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_arrow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_extension.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_masked.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_numpy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_sparse.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/test_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/uuid/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/extension/uuid/test_uuid.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/constructors/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/constructors/test_from_dict.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/constructors/test_from_records.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_coercion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_delitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_get.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_get_value.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_getitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_insert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_mask.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_set_value.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_setitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_where.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/indexing/test_xs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_add_prefix_suffix.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_align.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_asfreq.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_asof.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_assign.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_at_time.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_between_time.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_clip.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_combine.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_combine_first.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_compare.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_convert_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_copy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_count.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_cov_corr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_describe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_diff.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_dot.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_drop.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_drop_duplicates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_droplevel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_dropna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_duplicated.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_equals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_explode.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_filter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_first_valid_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_get_numeric_data.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_head_tail.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_infer_objects.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_info.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_interpolate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_is_homogeneous_dtype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_isetitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_isin.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_iterrows.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_map.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_matmul.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_nlargest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_pct_change.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_pipe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_pop.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_quantile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_rank.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_reindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_reindex_like.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_rename.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_rename_axis.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_reorder_levels.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_reset_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_round.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_sample.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_select_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_set_axis.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_set_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_shift.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_size.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_sort_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_sort_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_swaplevel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_csv.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_dict.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_dict_of_blocks.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_numpy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_records.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_to_timestamp.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_transpose.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_truncate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_tz_convert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_tz_localize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_update.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_value_counts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/methods/test_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_alter_axes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_arrow_interface.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_block_internals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_cumulative.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_iteration.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_logical_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_nonunique_indexes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_npfuncs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_query_eval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_repr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_stack_unstack.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_ufunc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_unary.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/frame/test_validate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_duplicate_labels.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_finalize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_frame.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_generic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_label_or_level_utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/generic/test_to_xarray.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/aggregate/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/aggregate/test_aggregate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/aggregate/test_cython.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/aggregate/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/aggregate/test_other.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_describe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_groupby_shift_diff.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_is_monotonic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_kurt.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_nlargest_nsmallest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_nth.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_quantile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_rank.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_sample.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_size.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_skew.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/methods/test_value_counts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_all_methods.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_apply.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_bin_groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_counting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_cumulative.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_filters.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_groupby_dropna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_groupby_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_grouping.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_index_as_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_libgroupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_numeric_only.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_pipe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_raises.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/test_timegrouper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/transform/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/transform/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/groupby/transform/test_transform.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_reshape.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/base_class/test_where.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_append.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_category.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_equals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_map.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_reindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/categorical/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_drop_duplicates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_equals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_is_monotonic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_nat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_sort_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimelike_/test_value_counts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_asof.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_delete.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_factorize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_insert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_isocalendar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_map.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_normalize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_repeat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_resolution.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_round.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_shift.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_snap.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_to_frame.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_to_julian_date.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_to_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_to_pydatetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_to_series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_tz_convert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_tz_localize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/methods/test_unique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_date_range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_freq_attr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_iter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_npfuncs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_partial_slicing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_reindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_scalar_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/datetimes/test_timezones.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_equals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_interval_range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_interval_tree.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_analytics.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_conversion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_copy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_drop.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_duplicates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_equivalence.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_get_level_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_get_set.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_integrity.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_isin.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_lexsort.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_monotonic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_names.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_partial_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_reindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_reshape.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_sorting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/multi/test_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/test_numeric.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/numeric/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/object/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/object/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/object/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_asfreq.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_factorize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_insert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_is_full.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_repeat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_shift.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/methods/test_to_timestamp.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_freq_attr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_monotonic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_partial_slicing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_period_range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_resolution.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_scalar_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_searchsorted.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/period/test_tools.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/test_range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/ranges/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/string/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/string/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/string/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_any_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_engines.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_frozen.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_index_new.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_numpy_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_old_base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/test_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_factorize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_insert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_repeat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/methods/test_shift.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_delete.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_freq_attr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_scalar_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_searchsorted.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_setops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_timedelta.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexes/timedeltas/test_timedelta_range.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/interval/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/interval/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/interval/test_interval_new.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_chaining_and_caching.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_getitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_iloc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_indexing_slow.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_loc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_multiindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_partial.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_setitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_slice.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/multiindex/test_sorted.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_at.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_chaining_and_caching.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_check_indexer.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_coercion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_floats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_iat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_iloc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_indexers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_loc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_na_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_partial.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/indexing/test_scalar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/interchange/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/interchange/test_impl.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/interchange/test_spec_conformance.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/interchange/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/internals/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/internals/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/internals/test_internals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_odf.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_odswriter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_openpyxl.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_readers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_style.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_writers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_xlrd.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/excel/test_xlsxwriter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_bar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_exceptions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_format.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_highlight.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_html.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_matplotlib.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_non_unique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_style.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_to_latex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_to_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_to_typst.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/style/test_tooltip.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_console.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_css.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_eng_formatting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_format.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_ipython_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_printing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_csv.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_excel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_html.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_latex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_markdown.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/formats/test_to_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/generate_legacy_storage_files.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_compression.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_deprecated_kwargs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_json_table_schema.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_json_table_schema_ext_dtype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_normalize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_pandas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_readlines.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/json/test_ujson.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_chunksize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_common_basic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_data_list.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_decimal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_file_buffer_url.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_float.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_inf.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_ints.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_iterator.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/common/test_read_errors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/dtypes/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/dtypes/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/dtypes/test_dtypes_basic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/dtypes/test_empty.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_c_parser_only.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_comment.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_compression.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_concatenate_chunks.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_converters.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_dialect.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_encoding.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_header.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_index_col.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_mangle_dupes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_multi_thread.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_na_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_network.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_parse_dates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_python_parser_only.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_quoting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_read_fwf.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_skiprows.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_textreader.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_unsupported.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/test_upcast.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/usecols/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/usecols/test_parse_dates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/usecols/test_strings.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/parser/usecols/test_usecols_basic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_append.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_compat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_complex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_errors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_file_handling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_keys.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_put.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_pytables_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_read.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_retain_attributes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_round_trip.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_select.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_store.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_time_series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/pytables/test_timezones.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/sas/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/sas/test_byteswap.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/sas/test_sas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/sas/test_sas7bdat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/sas/test_xport.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_clipboard.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_compression.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_feather.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_fsspec.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_gcs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_html.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_http_headers.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_iceberg.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_orc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_parquet.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_pickle.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_s3.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_spss.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_sql.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_stata.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/test_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/xml/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/xml/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/xml/test_to_xml.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/xml/test_xml.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/io/xml/test_xml_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/libs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/libs/test_hashtable.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/libs/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/libs/test_lib.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/libs/test_libalgos.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_frame.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_frame_color.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_frame_groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_frame_legend.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_frame_subplots.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/frame/test_hist_box_by.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_backend.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_boxplot_method.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_converter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_datetimelike.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_hist_method.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_misc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/plotting/test_style.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reductions/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reductions/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reductions/test_stat_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_base.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_datetime_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_period_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_resample_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_resampler_grouper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_time_grouper.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/resample/test_timedelta.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_append.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_append_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_categorical.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_concat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_dataframe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_datetimes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_empty.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_invalid.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_series.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/concat/test_sort.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_join.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge_antijoin.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge_asof.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge_cross.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge_index_as_string.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_merge_ordered.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/merge/test_multi.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_crosstab.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_cut.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_from_dummies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_get_dummies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_melt.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_pivot.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_pivot_multilevel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_qcut.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/reshape/test_union_categoricals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_contains.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_interval.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/interval/test_overlaps.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/period/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/period/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/period/test_asfreq.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/period/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/test_na_scalar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/test_nat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/methods/test_as_unit.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/methods/test_round.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timedelta/test_timedelta.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_as_unit.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_normalize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_round.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_timestamp_method.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_to_julian_date.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_to_pydatetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_tz_convert.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/methods/test_tz_localize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_comparisons.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_timestamp.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/scalar/timestamp/test_timezones.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_cat_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_dt_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_list_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_sparse_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_str_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/accessors/test_struct_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_delitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_get.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_getitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_indexing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_mask.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_set_value.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_setitem.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_where.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/indexing/test_xs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_add_prefix_suffix.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_align.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_argsort.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_asof.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_astype.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_autocorr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_between.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_case_when.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_clip.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_combine.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_combine_first.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_compare.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_convert_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_copy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_count.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_cov_corr.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_describe.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_diff.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_drop.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_drop_duplicates.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_dropna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_duplicated.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_equals.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_explode.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_fillna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_get_numeric_data.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_head_tail.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_infer_objects.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_info.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_interpolate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_is_monotonic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_is_unique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_isin.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_isna.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_item.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_map.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_matmul.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_nlargest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_nunique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_pct_change.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_pop.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_quantile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_rank.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_reindex.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_reindex_like.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_rename.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_rename_axis.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_repeat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_reset_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_round.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_searchsorted.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_set_name.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_size.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_sort_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_sort_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_to_csv.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_to_dict.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_to_frame.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_to_numpy.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_tolist.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_truncate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_tz_localize.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_unique.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_unstack.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_update.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_value_counts.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/methods/test_values.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_arithmetic.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_arrow_interface.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_constructors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_cumulative.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_formats.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_iteration.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_logical_ops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_missing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_npfuncs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_reductions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_subclass.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_ufunc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_unary.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/series/test_validate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_case_justify.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_cat.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_extract.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_find_replace.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_get_dummies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_split_partition.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_string_array.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/strings/test_strings.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_aggregation.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_algos.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_col.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_downstream.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_errors.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_expressions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_flags.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_multilevel.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_nanops.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_optional_dependency.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_register_accessor.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_sorting.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/test_take.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tools/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tools/test_to_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tools/test_to_numeric.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tools/test_to_time.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tools/test_to_timedelta.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/frequencies/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/frequencies/test_freq_code.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/frequencies/test_frequencies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/frequencies/test_inference.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/holiday/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/holiday/test_calendar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/holiday/test_federal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/holiday/test_holiday.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/holiday/test_observance.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_day.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_halfyear.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_hour.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_month.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_quarter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_business_year.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_common.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_custom_business_day.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_custom_business_hour.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_custom_business_month.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_dst.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_easter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_fiscal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_halfyear.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_index.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_month.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_offsets.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_offsets_properties.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_quarter.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_ticks.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_week.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tseries/offsets/test_year.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_array_to_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_ccalendar.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_conversion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_fields.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_libfrequencies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_liboffsets.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_np_datetime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_npy_units.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_parse_iso8601.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_parsing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_period.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_resolution.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_strptime.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_timedeltas.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_timezones.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_to_offset.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/tslibs/test_tzconversion.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_almost_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_attr_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_categorical_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_extension_array_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_frame_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_index_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_interval_array_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_numpy_array_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_produces_warning.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_assert_series_equal.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_deprecate.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_deprecate_kwarg.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_deprecate_nonkeyword_arguments.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_doc.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_hashing.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_rewrite_warning.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_shares_memory.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_show_versions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_util.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_validate_args.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_validate_args_and_kwargs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_validate_inclusive.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/util/test_validate_kwargs.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/moments/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/moments/conftest.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/moments/test_moments_consistency_ewm.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/moments/test_moments_consistency_expanding.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/moments/test_moments_consistency_rolling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_apply.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_base_indexer.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_cython_aggregations.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_ewm.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_expanding.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_groupby.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_numba.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_online.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_pairwise.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_rolling.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_rolling_functions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_rolling_quantile.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_rolling_skew_kurt.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_timeseries_window.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tests/window/test_win_type.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tseries/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tseries/api.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tseries/frequencies.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tseries/holiday.py
?? .venv_tf/lib/python3.12/site-packages/pandas/tseries/offsets.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_decorators.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_doctools.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_exceptions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_print_versions.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_test_decorators.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_tester.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/_validators.py
?? .venv_tf/lib/python3.12/site-packages/pandas/util/version/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pasta/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/errors.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/import_utils.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/import_utils_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/inline.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/inline_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/rename.py
?? .venv_tf/lib/python3.12/site-packages/pasta/augment/rename_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/__init__.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/annotate.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/annotate_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/ast_constants.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/ast_utils.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/ast_utils_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/codegen.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/codegen_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/formatting.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/fstring_utils.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/scope.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/scope_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/test_utils_test.py
?? .venv_tf/lib/python3.12/site-packages/pasta/base/token_generator.py
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/sboms/auditwheel.cdx.json
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/pillow-12.2.0.dist-info/zip-safe
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libXau-154567c4.so.6.0.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libavif-a883386a.so.16.4.1
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libbrotlicommon-785b7a00.so.1.2.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libbrotlidec-08d35d18.so.1.2.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libfreetype-d2d71ad9.so.6.20.6
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libharfbuzz-ef65cc8a.so.0.61321.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libjpeg-8296d2fa.so.62.4.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/liblcms2-545f6561.so.2.0.18
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/liblzma-3c942967.so.5.8.3
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libopenjp2-f3576751.so.2.5.4
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libpng16-65c953c0.so.16.56.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libsharpyuv-0bacc318.so.0.1.2
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libtiff-f4bc73ff.so.6.2.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libwebp-67ee3789.so.7.2.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libwebpdemux-eeb71312.so.2.0.17
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libwebpmux-20a90b09.so.3.1.2
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libxcb-bac6ebc8.so.1.1.0
?? .venv_tf/lib/python3.12/site-packages/pillow.libs/libzstd-6ea785c0.so.1.5.7
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/AUTHORS.txt
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/cachecontrol/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/certifi/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/distlib/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/distro/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/idna/LICENSE.md
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/msgpack/COPYING
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.APACHE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.BSD
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/pkg_resources/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/platformdirs/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/pygments/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/pyproject_hooks/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/requests/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/resolvelib/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/rich/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/tomli/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/tomli_w/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/truststore/LICENSE
?? .venv_tf/lib/python3.12/site-packages/pip-26.1.1.dist-info/licenses/src/pip/_vendor/urllib3/LICENSE.txt
?? .venv_tf/lib/pyth…627063 tokens truncated…ocation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/allocation_value.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/buffer_interval_comparator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/cost_analysis.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/memory_bound_loop_optimizer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/memory_space_assignment.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/memory_space_assignment.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/prefetch_interval_picker.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/repacking.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/simulator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/slice.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/tuning_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/memory_space_assignment/utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/metrics.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/metrics_hook_interface.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/multi_output_fusion.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/name_uniquer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/p2p_schedule_preparation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/pattern_matcher.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/platform_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/profile_guided_latency_estimator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/reduce_scatter_combiner.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/reduce_scatter_reassociate.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/rendezvous.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/riegeli_dump_writer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scan_expander.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scatter_determinism_expander.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scatter_expander.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scatter_simplifier.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scatter_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/schedule_config.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/scheduling_annotations_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/select_and_scatter_expander.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/service.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/service_executable_run_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/shape_inference.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/shaped_buffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/shaped_slice.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/shaped_slice.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/sharding_config.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/sharding_propagation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/sharding_remover.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/slow_operation_alarm.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/source_map_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/source_target_pairs.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/collective_permute_motion.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/convolution_handler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/custom_call_handler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/dot_handler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/schedule_aware_collective_ops_cse.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shard_barrier_partitioner.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/constants.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/extensions/mhlo_extensions.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/export_named_computations.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/import_constants.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/import_func_calls.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/import_sdy_custom_calls.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/round_trip_common/pipeline_passes.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/export_ops.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/pipelines.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/sdy_round_trip/shard_map_import.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/shardy_xla_pass.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/export_callback_custom_calls.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/export_manual_reduction_collectives.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/export_ops.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/shard_map_export.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/shard_map_import.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/shardy/utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/spmd_partitioner.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/spmd_partitioner_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/spmd/stateful_rng_spmd_partitioner.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/stream_pool.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/time_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/topk_rewriter.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/transfer_manager.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/transpose_folding.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/triangular_solve_expander.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/tuple_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/value_range.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_loop_all_reduce_code_motion.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_loop_constant_sinking.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_loop_invariant_code_motion.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_loop_simplifier.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_loop_unroller.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/while_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/service/xla_debug_info_manager.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_layout.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_partition.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_pool.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_tree.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/shape_util.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/sharding_op_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/side_effect_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/sort_json.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/status_macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/activate_context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/allocator_stats.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/bit_pattern.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/blas.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/blas.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/command_buffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/assemble_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/collective_signal_cuda.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/command_buffer_kernels.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/compilation_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/compilation_provider_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/composite_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cub_sort_kernel_cuda.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cubin_or_ptx_image.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_asm_compiler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_blas.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_blas_lt.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_blas_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_command_buffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_compute_capability.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_compute_capability.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_core_info_table.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_diagnostics.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_dnn.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_event.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_executor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_fft.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_helpers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_platform.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_platform_id.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_status.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_stream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_timer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cuda_version_parser.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cudnn_api_wrappers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cudnn_frontend_helpers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/cudnn_sdpa_score_mod.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/defer_relocatable_compilation_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/delay_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/driver_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nccl_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvjitlink.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvjitlink_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvjitlink_known_issues.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvjitlink_support.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvptxcompiler_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvshmem.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/nvshmem_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/ptx_compiler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/ptx_compiler_helpers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/ptx_compiler_support.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/subprocess_compilation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/subprocess_compilation_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/tma_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/cuda/topk_kernel_cuda_common.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/data_type.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_address.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_address_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_address_handle.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_description.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_description.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_memory.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/device_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/dnn.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/engine_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/event.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/event_based_timer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/executor_cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/fft.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/generic_memory_allocation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/generic_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/all_reduce_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/asm_compiler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/buffer_comparator_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/buffer_comparator_kernel_lib.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/buffer_debug_float_check_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/buffer_debug_log.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/buffer_debug_xor_checksum_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/collective_kernel_metadata.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/collective_signal.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/context_map.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_asm_opts.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_blas_lt.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_blas_lt.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_command_buffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_executor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_helpers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_init.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_kernel_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/gpu_semaphore.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/make_batch_pointers_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/multi_gpu_barrier_kernel.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/multi_gpu_barrier_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/multicast_memory.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/ragged_all_to_all_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/read_numa_node.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/redzone_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/redzone_allocator_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/redzone_allocator_kernel_lib.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/repeat_buffer_kernel.cu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/repeat_buffer_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/scoped_activate_context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/tma_metadata.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/tma_metadata.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu/topk_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/gpu_solver_context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_event.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_executor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_platform.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_platform_id.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_stream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/host/host_stream_factory.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/integrations/device_mem_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/integrations/stream_executor_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/integrations/tf_allocator_adapter.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_args.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_args_packed_vector.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_args_packing_spec.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_args_packing_spec.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_metadata.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_spec.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_spec.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_stats.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/kernel_symbol_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/launch_dim.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/launch_dim.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/lazy_op_runner.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/memory_allocation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/memory_space.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/module_spec.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/platform.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/platform/default/initialize.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/platform/initialize.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/platform/platform_object_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/platform_manager.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/plugin_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/rocm/rocm_compute_capability.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/rocm/rocm_platform_id.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/scoped_module_handle.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/scratch_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/semantic_version.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream_common.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream_executor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream_executor_address_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream_executor_common.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/stream_executor_memory_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/sycl/oneapi_compute_capability.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/sycl/oneapi_compute_capability.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/sycl/sycl_platform_id.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tensor_map.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/c_api_conversions.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/c_api_decl.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/c_api_defn.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/libtftpu.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/proto_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/tpu_api.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/tpu_executor_api.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/tpu_executor_c_api.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/tpu_ops_c_api.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/tpu/tpu_profiler_c_api.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/trace_command_buffer_factory.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/stream_executor/typed_kernel_factory.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tools/hlo_decomposer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tools/hlo_extractor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/c/tsl_status.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/c/tsl_status_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/c/tsl_status_internal.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/async_value.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/async_value_ref.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/chain.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/concurrent_vector.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/executor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/future.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/concurrency/ref_count.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cublas.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cublasLt.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cuda.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cudart.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cudnn.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cufft.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cupti.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cusolver.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/cusparse.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/nccl.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/nvml.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/cuda/nvshmem.inc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/call_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/coordination_client.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/coordination_service.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/coordination_service_agent.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/coordination_service_rpc_handler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/coordination/key_value_store.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/preemption/preemption_notifier.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/async_service_interface.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_call.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_channel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_channel_common.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_client_cq_tag.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_state.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/distributed_runtime/rpc/grpc_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/allocator_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/allocator_retry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/bfc_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/cancellation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/contraction/eigen_contraction_kernel.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/convolution/eigen_convolution_helpers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/convolution/eigen_spatial_convolutions.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/device_id.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/device_id_manager.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/device_id_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/device_type.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/FixedPoint.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/MatMatProduct.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/MatMatProductAVX2.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/MatMatProductNEON.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/MatVecProduct.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/PacketMathAVX.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/PacketMathAVX2.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/PacketMathAVX512.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/TypeCastingAVX2.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint/TypeCastingAVX512.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/fixedpoint_types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/metrics.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/numeric_types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/real_time_in_memory_metric.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/serving_device_selector.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/serving_device_selector_policies.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/shared_counter.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/tracking_allocator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/framework/type_traits.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/core/bitmap.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/core/bits.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/core/status_test_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/compactptrset.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/flatmap.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/flatrep.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/flatset.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/inlined_vector.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/int_type.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/iterator_range.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/map_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/subtle/map_traits.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/gtl/value_or_die.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/hash/crc32c.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/histogram/histogram.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/block.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/block_builder.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/buffered_file.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/buffered_inputstream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/compression.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/format.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/inputbuffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/inputstream_interface.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/iterator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/proto_encode_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/random_inputstream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/record_reader.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/record_writer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/snappy/snappy_compression_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/snappy/snappy_inputbuffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/snappy/snappy_inputstream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/snappy/snappy_outputbuffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/table.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/table_builder.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/table_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/two_level_iterator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/zlib_compression_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/zlib_inputstream.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/io/zlib_outputbuffer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/math/math_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/cell_reader.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/collected_metrics.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/collection_registry.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/counter.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/gauge.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/metric_def.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/percentile_sampler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/sampler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/test_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/timed.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/monitoring/types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/distribution_sampler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/exact_uniform_int.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/philox_random.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/philox_random_test_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/random_distributions.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/random_distributions_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/simple_philox.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/random/weighted_picker.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/lib/strings/proto_serialization.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/byte_order.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/auth_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/compute_engine_metadata_client.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/compute_engine_zone_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/curl_http_request.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/expiring_lru_cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/file_block_cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/gcs_dns_cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/gcs_file_system.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/gcs_throttle.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/google_auth_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/http_request.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/oauth_client.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/ram_file_block_cache.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/time_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/cloud/zone_provider.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/crash_analysis.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/criticality.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/debug_me_context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/casts.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/crash_analysis.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/criticality.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/dso_loader.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/posix_file_system.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/stacktrace.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/status.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/statusor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/subprocess.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/tracing_impl.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/default/unbounded_work_queue.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/dynamic_annotations.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/env.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/env_time.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/errors.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/file_statistics.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/file_system.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/file_system_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/grpc_credentials.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/intrusive_ptr.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/logging.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/prefetch.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/profile_utils/android_armv7a_cpu_utils_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/profile_utils/clock_cycle_profiler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/profile_utils/cpu_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/profile_utils/i_cpu_utils_helper.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/protobuf_compiler.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/ram_file_system.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/resource.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/resource_loader.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/rocm_rocdl_path.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/stack_frame.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/status.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/status_macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/status_matchers.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/status_to_from_proto.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/statusor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/subprocess.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/test.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/test_benchmark.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/threadpool.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/threadpool_interface.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/threadpool_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/platform/types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/backends/cpu/annotation_stack.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/backends/cpu/host_tracer_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/backends/cpu/threadpool_listener.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/backends/cpu/threadpool_listener_state.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/backends/cpu/traceme_recorder.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/convert/post_process_single_host_xplane.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/convert/xla_op_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/rpc/client/save_profile.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/rpc/profiler_service_impl.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/buffer_pool.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/file_system_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/lock_free_queue.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/math_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/no_init.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/parse_annotation.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/per_thread.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/profiler_options_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/tf_op_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/tf_xplane_visitor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/time_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/timespan.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/timestamp_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/trace_filter_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/trace_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/xplane_builder.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/xplane_schema.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/xplane_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/profiler/utils/xplane_visitor.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/bfc_memory_map.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/coordination_config.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/coordination_service.grpc.pb.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/coordination_service.grpc.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/coordination_service.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/coordination_service_mock.grpc.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/distributed_runtime_payloads.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/dnn.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/error_codes.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/histogram.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/rpc_options.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/status.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/protobuf/test_log.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/python/lib/core/ml_dtypes.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/python/lib/core/numpy.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/byte_swap_array.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/command_line_flags.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/determinism.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/device_name_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/env_var.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/onednn_threadpool.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/proto/proto_utils.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/reporter.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/safe_reinterpret_cast.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/stat_summarizer_options.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/stats_calculator.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tsl/util/use_cudnn.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/tuple_tree.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/union_find.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/util/split_proto/split_executable_and_options_writer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/util/split_proto/split_gpu_executable_writer.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/util/split_proto/split_proto.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/util/split_proto/split_proto_reader.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/window_util.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/xla.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/include/xla/xla_data.pb.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/libtensorflow_cc.so.2
?? .venv_tf/lib/python3.12/site-packages/tensorflow/libtensorflow_framework.so.2
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/genai/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/genai/pywrap_genai_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/ops/gen_audio_microfrontend_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/python/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/python/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/experimental/microfrontend/python/ops/audio_microfrontend_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/profiling/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/profiling/proto/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/profiling/proto/model_runtime_info_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/profiling/proto/profiling_info_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_analyzer_wrapper.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_modify_model_interface.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_string_util.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_tensorflow_interpreter_wrapper.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_tensorflow_lite_calibration_wrapper.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/_pywrap_tensorflow_lite_metrics_wrapper.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/analyzer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/analyzer_wrapper/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/analyzer_wrapper/_pywrap_analyzer_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/authoring/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/authoring/authoring.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/conversion_metadata_schema_py_generated.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/convert.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/convert_phase.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/convert_saved_model.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/format_converter_wrapper_pybind11.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/interpreter_wrapper/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/lib_pywrap_tflite_7_shared_object.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/libpywrap_tflite_common.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/lite.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/lite_constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/_pywrap_tensorflow_lite_metrics_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/_pywrap_tensorflow_lite_metrics_wrapper.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/metrics.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/metrics_interface.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/wrapper/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/metrics/wrapper/metrics_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/op_hint.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/optimize/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/optimize/_pywrap_tensorflow_lite_calibration_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/optimize/_pywrap_tensorflow_lite_calibration_wrapper.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/optimize/calibrator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/pywrap_genai_ops.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/schema_py_generated.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/schema_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/tflite_convert.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/tflite_keras_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/python/util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/testing/_pywrap_string_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/logging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/logging/gen_html.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/logging/toco_conversion_log_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/toco_flags_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/toco/types_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/flatbuffer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/debugging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/debugging/python/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/debugging/python/debugger.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/python/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/python/_pywrap_modify_model_interface.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/sparsity/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/optimize/sparsity/format_converter_wrapper_pybind11.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/lite/tools/visualize.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_dtensor_device.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_dtensor_device.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_mlir.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_mlir.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_parallel_device.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_parallel_device.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_py_exception_registry.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_py_exception_registry.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_quantize_training.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_quantize_training.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_sanitizers.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_sanitizers.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_tfcompile.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_tfcompile.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_tfe.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_tfe.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/_pywrap_toco_api.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/asserts.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/break_statements.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/call_trees.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/conditional_expressions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/continue_statements.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/control_flow.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/directives.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/functions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/lists.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/logical_expressions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/return_statements.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/slices.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/converters/variables.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/ag_ctx.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/config_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/converter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/converter_testing.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/function_wrappers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/core/unsupported_features_checker.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/api.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/conversion.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/testing/pybind_for_testing.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/impl/testing/pybind_for_testing.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/lang/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/lang/directives.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/lang/special_functions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/conditional_expressions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/control_flow.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/data_structures.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/exceptions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/logical.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/py_builtins.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/slices.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/operators/variables.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/anno.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/ast_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/cache.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/cfg.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/common_transformers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/common_transformers/anf.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/error_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/errors.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/gast_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/inspect_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/loader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/naming.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/origin_info.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/parser.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/pretty_printer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/qual_names.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/activity.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/annos.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/reaching_definitions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/reaching_fndefs.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/static_analysis/type_inference.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/templates.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/testing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/testing/basic_definitions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/testing/decorators.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/transformer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/pyct/transpiler.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/ag_logging.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/context_managers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/misc.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/tensor_list.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/tensors.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/testing.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/autograph/utils/type_registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/async_checkpoint_helper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint_adapter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint_management.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint_options.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/checkpoint_view.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/functional_saver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/graph_view.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/restore.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/save_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/save_util_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/saveable_compat.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/sharding/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/sharding/sharding_policies.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/sharding/sharding_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/tensor_callable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/trackable_view.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/checkpoint/util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_debug_events_writer.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_debug_events_writer.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_device_lib.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_device_lib.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_events_writer.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_events_writer.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_tf_session.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/_pywrap_tf_session.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/client_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/device_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/pywrap_tf_session.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/session.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/client/timeline.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compat/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compat/compat.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compat/v2_compat.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/mlir/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/mlir/mlir.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/tensorrt/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/tensorrt/test/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/tensorrt/test/tf_trt_integration_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/tensorrt/trt_convert.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/tensorrt/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/xla/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/xla/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/xla/experimental/xla_sharding.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/xla/jit.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/compiler/xla/xla.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/benchmarks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/benchmarks/benchmark_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/kernel_tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/kernel_tests/service/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/kernel_tests/service/multi_process_cluster.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/kernel_tests/service/test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/batching.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/cardinality.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/compression_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/counter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/data_service_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/distribute.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/distributed_save_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/enumerate_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/error_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/from_list.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/get_single_element.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/global_shuffle_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/grouping.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/interleave_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/io.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/iterator_model_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/iterator_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/lookup_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/map_defun.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/matching_files.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/pad_to_cardinality.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/parsing_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/prefetching_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/random_access.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/readers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/resampling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/scan_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/shuffle_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/snapshot.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/take_while_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/testing.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/unique.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/ops/writers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_server_lib.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_server_lib.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_snapshot_utils.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_snapshot_utils.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_utils_exp.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/_pywrap_utils_exp.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/experimental/service/server_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/kernel_tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/kernel_tests/checkpoint_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/kernel_tests/test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/kernel_tests/tf_record_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/batch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/cache_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/choose_from_datasets_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/concatenate_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/counter_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/dataset_autograph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/dataset_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/debug_mode.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/directed_interleave_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/filter_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/flat_map_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/from_generator_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/from_sparse_tensor_slices_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/from_tensor_slices_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/from_tensors_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/group_by_window_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/ignore_errors_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/interleave_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_autograph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/load_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/map_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/optional_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/options.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/padded_batch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/prefetch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/ragged_batch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/random_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/range_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/readers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/rebatch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/repeat_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/sample_from_datasets_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/save_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/scan_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/shard_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/shuffle_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/skip_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/snapshot_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/sparse_batch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/structured_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/take_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/take_while_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/test_mode.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/unbatch_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/unique_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/window_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/ops/zip_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/convert.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/nest.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/options.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/random_seed.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/sparse.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/structure.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/data/util/traverse.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/analyzer_cli.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/base_ui.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/cli_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/cli_shared.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/cli_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/command_parser.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/debugger_cli_common.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/evaluator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/offline_analyzer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/profile_analyzer_cli.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/readline_ui.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/tensor_format.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/cli/ui_factory.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/check_numerics_callback.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/common.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_data.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_events_monitors.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_events_reader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_events_writer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_gradients.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_graphs.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_service_pb2_grpc.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/debug_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/dumping_callback.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/dumping_callback_test_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/grpc_debug_server.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/grpc_debug_test_server.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/grpc_tensorflow_server.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/op_callbacks_common.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/profiling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/session_debug_testlib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/source_remote.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/lib/source_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/dumping_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/framework.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/grpc_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/hooks.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/debug/wrappers/local_cli_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/central_storage_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/gce_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/kubernetes_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/sagemaker_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/tfconfig_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/tpu/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cluster_resolver/tpu_cluster_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/collective_all_reduce_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/collective_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/combinations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/cluster_coordinator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/coordinator_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/fault_tolerance_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/metric_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/remote_value.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/values.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/coordinator/watchdog.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cross_device_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/cross_device_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/device_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/distribute_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/distribute_coordinator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/distribute_coordinator_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/distribute_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/distribute_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/dtensor_strategy_extended.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/dtensor_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/mirrored_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/multi_worker_mirrored_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/rpc/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/experimental/rpc/rpc_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/failure_handling/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/failure_handling/failure_handling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/failure_handling/failure_handling_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/failure_handling/preemption_watcher.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/input_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/input_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/input_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/load_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/merge_call_interim.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/mirrored_run.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/mirrored_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/multi_process_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/multi_process_runner.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/multi_worker_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/multi_worker_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/numpy_dataset.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/one_device_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/packed_distributed_variable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/parallel_device/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/parallel_device/parallel_device.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/parameter_server_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/parameter_server_strategy_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/ps_values.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/reduce_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/sharded_variable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/shared_variable_creator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/single_loss_example.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/step_fn.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/strategy_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/strategy_test_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/summary_op_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/tpu_replicated_variable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/tpu_strategy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/tpu_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/tpu_values.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/v1/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/v1/all_reduce.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/v1/input_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/values.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/values_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/distribute/values_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/dlpack/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/dlpack/dlpack.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/backprop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/backprop_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/benchmarks_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/cancellation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/core.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/def_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/execute.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/executor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/forwardprop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/forwardprop_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/graph_only_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/imperative_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/lift_to_graph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/memory_tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/memory_tests/memory_test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/monitoring.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/attributes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/autograph_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/compiler_ir.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/composite_tensor_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/eager_function_run.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/function_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/function_type_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/saved_model_exported_concrete.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/saved_model_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tf_method_target.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/transform.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/record.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/remote.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/tape.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/test.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/wrap_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/feature_column.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/feature_column_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/feature_column_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/feature_column_v2_types.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/sequence_feature_column.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/feature_column/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/flags_pybind.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/flags_pybind.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_dtypes.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_dtypes.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_op_def_library_pybind.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_op_def_library_pybind.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_op_def_registry.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_op_def_registry.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_proto_comparators.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_proto_comparators.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_python_memory_checker_helper.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_python_memory_checker_helper.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_pywrap_python_api_dispatcher.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_pywrap_python_op_gen.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_pywrap_python_op_gen.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_test_metrics_util.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/_test_metrics_util.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/auto_control_deps.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/auto_control_deps_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/byte_swap_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/c_api_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/combinations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/common_shapes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/composite_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/composite_tensor_gradient.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/constant_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/constant_tensor_conversion.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/convert_to_constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/cpp_shape_inference_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/device.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/device_spec.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/dtypes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/error_interpolation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/errors.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/errors_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/extension_type.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/extension_type_field.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/flexible_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/framework_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/func_graph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/function_def_to_graph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/gpu_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/graph_io.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/graph_to_function_def.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/graph_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/graph_util_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/immutable_dict.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/importer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/indexed_slices.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/kernels.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/lib_native_proto_caster.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/load_library.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/memory_checker.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/meta_graph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/none_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/op_callbacks.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/op_def_library.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/op_def_library_pybind.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/op_def_registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/override_binary_operator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/python_memory_checker.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/random_seed.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/smart_cond.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/sparse_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/stack.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/strict_mode.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/subscribe.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor_conversion.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor_conversion_registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor_shape.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor_spec.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tensor_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/test_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/test_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/tfrt_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/traceable_stack.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/type_spec.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/type_spec_registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/type_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/versions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/framework/weak_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_cluster.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_cluster.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_item.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_item.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_optimizer.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/_pywrap_tf_optimizer.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/cluster.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/item.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/grappler/tf_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/activations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/backend.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/backend_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/callbacks.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/callbacks_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/combinations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/constraints.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/distribute_coordinator_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/distributed_file_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/distributed_training_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/distributed_training_utils_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/distribute/worker_training_state.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/base_layer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/base_layer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/base_layer_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/base_preprocessing_layer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/compile_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/data_adapter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/functional.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/input_layer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/input_spec.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/keras_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/node.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/partial_batch_padding_handler.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/saving.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/sequential.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_arrays_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_distributed_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_eager_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_generator_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_utils_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/engine/training_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/initializers/initializers_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/initializers/initializers_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/keras_parameterized.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/advanced_activations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/convolutional.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/convolutional_recurrent.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/core.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/dense_attention.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/embeddings.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/legacy_rnn/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_wrapper_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/merge.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/pooling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/recurrent.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/rnn_cell_wrapper_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/layers/serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/convolutional.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/core.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/pooling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/legacy_tf_layers/variable_scope_shim.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/losses.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/metrics.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/autocast_variable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/device_compatibility_check.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/get_layer_policy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/loss_scale.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/loss_scale_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/policy.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/mixed_precision/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/models.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/adadelta.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/adagrad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/adam.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/adamax.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/ftrl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/gradient_descent.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/legacy_learning_rate_decay.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/nadam.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizer_v2/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/optimizers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/protobuf/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/protobuf/projector_config_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/protobuf/saved_metadata_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/protobuf/versions_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/regularizers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/hdf5_format.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/model_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/save.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/base_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/json_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/layer_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/load.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/load_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/metric_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/model_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/network_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/save.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/save_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/serialized_attributes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saved_model_experimental.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/saving_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/export_output.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/export_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/mode_keys.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/signature_def_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/saving/utils_v1/unexported_constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/testing_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/all_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/control_flow_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/conv_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/data_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/dataset_creator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/generic_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/io_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/kernelized_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/layer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/losses_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/metrics_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/mode_keys.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/np_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/object_identity.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/tf_contextlib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/tf_inspect.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/tf_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/version_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/keras/utils/vis_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/nn_ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/nn_ops/bias_op_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/nn_ops/cudnn_deterministic_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/nn_ops/depthwise_conv_op_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/nn_ops/xent_op_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/random/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/random/util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/signal/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/signal/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/sparse_ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/kernel_tests/sparse_ops/sparse_xent_op_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/convolutional.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/core.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/layers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/normalization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/pooling.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/layers/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/core/_pywrap_py_func.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/core/_pywrap_py_func.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/_pywrap_file_io.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/_pywrap_file_io.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/_pywrap_record_io.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/_pywrap_record_io.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/file_io.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/python_io.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib/io/tf_record.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/lib_pywrap_tensorflow_common.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/module/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/module/module.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/array_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/array_ops_stack.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/autograph_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/batch_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/bincount_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/bitwise_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/boosted_trees_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/candidate_sampling_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/check_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/clip_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/clustering_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/collective_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/composite_tensor_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/cond.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/cond_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/confusion_matrix.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_assert.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_case.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_state.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_switch_case.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_util_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_v2_func_graphs.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/control_flow_v2_toggles.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/critical_section_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ctc_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/cudnn_rnn_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/custom_gradient.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/data_flow_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/data_flow_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/default_gradient.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/bernoulli.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/beta.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/bijector.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/bijector_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/bijector_test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/categorical.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/dirichlet.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/dirichlet_multinomial.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/distribution.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/distributions.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/exponential.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/gamma.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/identity_bijector.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/kullback_leibler.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/laplace.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/multinomial.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/normal.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/special_math.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/student_t.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/uniform.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/distributions/util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/embedding_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/filesystem_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/functional_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_audio_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_batch_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_bitwise_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_boosted_trees_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_candidate_sampling_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_checkpoint_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_clustering_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_collective_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_composite_tensor_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_control_flow_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_count_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_ctc_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_cudnn_rnn_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_data_flow_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_dataset_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_debug_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_decode_proto_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_encode_proto_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_experimental_dataset_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_filesystem_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_functional_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_image_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_io_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_linalg_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_list_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_logging_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_lookup_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_manip_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_map_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_nccl_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_nn_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_optional_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_parsing_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_ragged_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_ragged_conversion_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_ragged_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_random_index_shuffle_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_resource_variable_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_rnn_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_script_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_sdca_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_sendrecv_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_set_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_sparse_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_special_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_spectral_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_state_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_stateful_random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_stateless_random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_stateless_random_ops_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_string_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_summary_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_sync_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_tpu_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_tpu_partition_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_training_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gen_uniform_quant_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gradient_checker.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gradient_checker_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gradients.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gradients_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/gradients_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/handle_data_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/histogram_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/image_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/image_grad_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/image_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/image_ops_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/init_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/init_ops_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/initializers_ns.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/inplace_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/io_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linalg.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linalg_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_addition.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_adjoint.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_block_diag.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_block_lower_triangular.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_circulant.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_composition.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_diag.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_full_matrix.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_householder.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_identity.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_inversion.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_kronecker.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_low_rank_update.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_permutation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_toeplitz.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_tridiag.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/linear_operator_zeros.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/property_hint_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/slicing.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/conjugate_gradient.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/gen_sparse_csr_matrix_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/sparse.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/sparse_csr_matrix_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg/sparse/sparse_csr_matrix_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/linalg_ops_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/list_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/logging_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/lookup_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/lookup_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/losses/losses.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/losses/losses_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/losses/util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/manip_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/manip_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/map_fn.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/map_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/math_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/metrics.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/metrics_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nccl_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn_fused_batch_norm_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn_impl_distribute.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/nn_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numerics.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_arrays.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_dtypes.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_random.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/numpy_ops/np_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/op_selector.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/optional_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parallel_for/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parallel_for/control_flow_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parallel_for/gradients.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parallel_for/pfor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parallel_for/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parsing_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parsing_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/parsing_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/partitioned_variables.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/proto_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/dynamic_ragged_shape.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_autograph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_batch_gather_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_batch_gather_with_default_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_bincount_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_check_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_concat_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_conversion_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_dispatch.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_embedding_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_factory_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_functional_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_gather_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_getitem.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_image_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_map_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_operators.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_squeeze_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_string_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_tensor_shape.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_tensor_test_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_tensor_value.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/ragged_where_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/row_partition.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ragged/segment_id_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/random_crop_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/random_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/random_ops_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/ref_variable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/resource_variable_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/resource_variables_toggle.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/resources.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/rnn.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/rnn_cell.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/rnn_cell_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/rnn_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sdca_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/session_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sets.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sets_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/shape_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/dct_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/fft_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/mel_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/mfcc_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/reconstruction_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/shape_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/signal.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/spectral_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/util_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/signal/window_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sort_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sparse_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/sparse_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/special_math_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/standard_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/state_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/state_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/stateful_random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/stateless_random_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/string_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/structured/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/structured/structured_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/structured/structured_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/structured/structured_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/structured/structured_tensor_dynamic.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/summary_op_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/summary_ops_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/template.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/tensor_array_grad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/tensor_array_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/tensor_getitem_override.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/tensor_math_operator_overrides.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/unconnected_gradients.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/variable_scope.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/variable_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/variables.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/weak_tensor_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/weak_tensor_test_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/weights_broadcast_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/while_loop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/while_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/ops/while_v2_indexed_slices_rewriter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_cpu_feature_guard.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_cpu_feature_guard.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_stacktrace_handler.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_stacktrace_handler.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_tf2.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/_pywrap_tf2.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/analytics.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/app.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/benchmark.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/build_info.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/device_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/flags.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/gfile.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/googletest.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/remote_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/resource_loader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/self_check.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/sysconfig.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/test.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/platform/tf_logging.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler_plugin.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler_plugin.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_traceme.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/_pywrap_traceme.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/internal/flops_registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/model_analyzer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/option_builder.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/profiler.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/profiler_client.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/profiler_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/tfprof_logger.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/profiler/trace.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/proto_exports.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/pywrap_dlopen_global_flags.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/pywrap_mlir.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/pywrap_sanitizers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/pywrap_tensorflow.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/pywrap_tfe.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/builder.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/builder_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/fingerprinting.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/fingerprinting_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/function_deserialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/function_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load_options.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load_v1_in_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/loader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/loader_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/main_op.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/main_op_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/method_name_updater.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/model_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/model_utils/export_output.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/model_utils/export_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/model_utils/mode_keys.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/nested_structure_coder.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/path_helpers.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model/constants.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model/fingerprinting.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model/merger.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/pywrap_saved_model/metrics.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/registration/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/registration/registration.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/revived_types.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/save.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/save_context.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/save_options.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/saved_model.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/signature_constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/signature_def_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/signature_def_utils_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/signature_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/simple_save.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/tag_constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/tracing_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/utils_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/plugin_asset.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/summary.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/summary_iterator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/tb_summary.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/writer/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/writer/event_file_writer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/writer/event_file_writer_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/writer/writer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/summary/writer/writer_cache.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tf2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator/create_python_api.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator/doc_srcs.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/extractor/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/extractor/extractor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/generator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/generator/generator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/shared/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/api/generator2/shared/exported_api.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/freeze_graph.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/import_pb_to_tensorboard.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/inspect_checkpoint.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/module_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/optimize_for_inference.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/optimize_for_inference_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/print_selective_registration_header.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/saved_model_aot_compile.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/saved_model_cli.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/saved_model_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/selective_registration_header_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/strip_unused.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tools/strip_unused_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/_pywrap_sparse_core_layout.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/_pywrap_sparse_core_layout.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/_pywrap_tpu_embedding.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/_pywrap_tpu_embedding.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/api.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/bfloat16.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/client/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/client/client.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/client/version.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/datasets.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/device_assignment.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/embedding_context_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/feature_column.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/feature_column_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/functional.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/ops/gen_xla_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/ops/tpu_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/preempted_hook.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tensor_tracer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tensor_tracer_flags.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tensor_tracer_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tensor_tracer_report.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/topology.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_for_serving.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v2_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v3.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v3_checkpoint_adapter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_embedding_v3_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_feed.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_function.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_hardware_feature.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_name_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_replication.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_sharding.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_strategy_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/tpu_system_metadata.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/tpu/training_loop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/asset.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/autotrackable.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/base.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/base_delegate.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/constants.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/converter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/data_structures.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/layer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/python_state.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/resource.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/trackable/trackable_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/adadelta.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/adagrad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/adagrad_da.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/adam.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/basic_loops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/basic_session_run_hooks.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/checkpoint_management.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/checkpoint_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/checkpoint_state_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/checkpoint_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/coordinator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/device_setter.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/evaluation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/experimental/loss_scale.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/experimental/loss_scale_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/experimental/mixed_precision.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/experimental/mixed_precision_global_state.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/ftrl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/gradient_descent.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/input.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/learning_rate_decay.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/momentum.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/monitored_session.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/moving_averages.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/proximal_adagrad.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/proximal_gradient_descent.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/py_checkpoint_reader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/quantize_training.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/queue_runner.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/queue_runner_impl.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saver_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saving/saveable_object.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saving/saveable_object_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/saving/trace_saveable_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/server_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/session_manager.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/session_run_hook.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/slot_creator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/summary_io.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/supervisor.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/sync_replicas_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/training.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/training_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/training/warm_starting_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/core.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/data.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/distribute.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/doc_typealias.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/internal.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/types/trace.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/user_ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/user_ops/ops/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/user_ops/ops/gen_user_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/user_ops/user_ops.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_checkpoint_reader.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_checkpoint_reader.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_determinism.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_determinism.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_kernel_registry.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_kernel_registry.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_nest.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_nest.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_stat_summarizer.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_stat_summarizer.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_tensor_float_32_execution.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_tensor_float_32_execution.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_tfprof.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_tfprof.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_transform_graph.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_transform_graph.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_util_port.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_util_port.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_utils.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_pywrap_utils.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_tf_stack.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/_tf_stack.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/all_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/compat.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/custom_nest_protocol.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/decorator_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/deprecated_module.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/deprecated_module_new.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/deprecation.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/dispatch.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/example_parser_configuration.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/fast_module_type.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/fast_module_type.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/function_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/is_in_graph_mode.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/keras_deps.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/keyword_args.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/lazy_loader.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/lock_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/module_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/nest.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/nest_util.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/numpy_compat.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/object_identity.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/protobuf/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/protobuf/compare.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/pywrap_xla_ops.pyi
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/pywrap_xla_ops.so
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/serialization.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_contextlib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_decorator.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_decorator_export.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_export.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_inspect.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_should_use.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/tf_stack.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/traceback_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/type_annotations.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/python/util/variable_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/security/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/security/fuzzing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/security/fuzzing/py/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/security/fuzzing/py/annotation_types.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/common/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/common/public_api.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/common/test_module1.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/common/test_module2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/common/traverse.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/all_renames_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/ast_edits.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/ipynb.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/module_deprecations_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/renames_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/reorders_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/tf_upgrade_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/tf_upgrade_v2_main.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/compatibility/tf_upgrade_v2_safety.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/docs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/docs/doc_controls.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/docs/tf_doctest_lib.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/pip_package/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tools/pip_package/setup.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/profile_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/profiler_options_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/trace_events_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/xplane_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/CMakeLists.txt
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/compiler/tf2xla/allocator.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/cord.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/ctstring.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/ctstring_internal.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/dynamic_annotations.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/env_time.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/platform.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/tstring.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tensorflow/core/platform/types.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/ctstring.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/ctstring_internal.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/env_time.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/platform.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/convolution_lib_f16_2d.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/convolution_lib_f16_3d.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/convolution_lib_f32_2d.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/convolution_lib_f32_3d.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_c128.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_c64.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_f16.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_f32.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_f64.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_s32.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/dot_lib_s8.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/rng_state_lib.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/backends/cpu/runtime/sort_lib.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/executable_run_options.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/service/custom_call_status.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/concurrency/async_value.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/concurrency/async_value_ref.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/framework/contraction/eigen_contraction_kernel.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/platform/default/context.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/platform/default/env_time.cc
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/platform/dynamic_annotations.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/platform/env_time.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow/xla_aot_runtime_src/xla/tsl/platform/macros.h
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/REQUESTED
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub-0.16.1.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/compressed_module_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/config.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/file_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/keras_layer.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/module_v2.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/registry.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/tf_utils.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/uncompressed_module_resolver.py
?? .venv_tf/lib/python3.12/site-packages/tensorflow_hub/version.py
?? .venv_tf/lib/python3.12/site-packages/termcolor-3.3.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/termcolor-3.3.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/termcolor-3.3.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/termcolor-3.3.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/termcolor-3.3.0.dist-info/licenses/COPYING.txt
?? .venv_tf/lib/python3.12/site-packages/termcolor/__init__.py
?? .venv_tf/lib/python3.12/site-packages/termcolor/__main__.py
?? .venv_tf/lib/python3.12/site-packages/termcolor/py.typed
?? .venv_tf/lib/python3.12/site-packages/termcolor/termcolor.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras-2.21.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/tf_keras-2.21.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/tf_keras-2.21.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/tf_keras-2.21.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/tf_keras-2.21.0.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/__internal__/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/legacy/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/legacy/layers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/__internal__/legacy/rnn_cell/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/convnext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/densenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/efficientnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/efficientnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/imagenet_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/inception_resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/inception_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/mobilenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/mobilenet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/mobilenet_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/nasnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/regnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/resnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/resnet50/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/resnet_rs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/vgg16/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/vgg19/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/applications/xception/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/callbacks/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/estimator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/layers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/layers/experimental/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/preprocessing/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/preprocessing/sequence/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/preprocessing/text/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v1/keras/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/__internal__/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/activations/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/convnext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/densenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/efficientnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/efficientnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/imagenet_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/inception_resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/inception_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/mobilenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/mobilenet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/mobilenet_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/nasnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/regnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/resnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/resnet50/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/resnet_rs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/vgg16/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/vgg19/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/applications/xception/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/backend/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/callbacks/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/dtensor/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/dtensor/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/dtensor/experimental/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/estimator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/layers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/layers/experimental/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/metrics/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/models/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/optimizers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/preprocessing/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/preprocessing/sequence/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/preprocessing/text/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/utils/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/api/_v2/keras/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/convnext/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/densenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/efficientnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/efficientnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/imagenet_utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/inception_resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/inception_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/mobilenet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/mobilenet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/mobilenet_v3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/nasnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/regnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/resnet/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/resnet50/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/resnet_rs/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/resnet_v2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/vgg16/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/vgg19/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/applications/xception/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/backend/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/backend/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/callbacks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/callbacks/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/constraints/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/dtensor/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/dtensor/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/dtensor/experimental/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/estimator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/layers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/layers/experimental/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/losses/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/metrics/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/models/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/optimizers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/preprocessing/image/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/preprocessing/sequence/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/preprocessing/text/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/protobuf/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/protobuf/projector_config_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/protobuf/saved_metadata_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/protobuf/versions_pb2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/regularizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/activations.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/convnext.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/densenet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/efficientnet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/efficientnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/imagenet_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/inception_resnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/inception_v3.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/mobilenet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/mobilenet_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/mobilenet_v3.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/nasnet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/regnet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/resnet.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/resnet_rs.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/resnet_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/vgg16.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/vgg19.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/applications/xception.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/backend.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/backend_config.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/benchmarks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/benchmarks/benchmark_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/benchmarks/distribution_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/benchmarks/model_memory_profile.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/callbacks.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/callbacks_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/constraints.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/dataset_creator_model_fit_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/distribute_coordinator_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/distributed_file_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/distributed_training_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/distributed_training_utils_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/keras_correctness_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/model_collection_base.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/model_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/multi_worker_testing_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/optimizer_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/saved_model_test_base.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/simple_models.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/strategy_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/test_example.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/tpu_strategy_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/distribute/worker_training_state.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/integration_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/layout_map.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/lazy_variable.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/dtensor/utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/base_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/base_layer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/base_layer_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/base_preprocessing_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/compile_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/data_adapter.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/functional.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/functional_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/input_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/input_spec.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/keras_tensor.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/node.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/partial_batch_padding_handler.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/saving.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/sequential.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_arrays_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_distributed_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_eager_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_generator_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_utils_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/engine/training_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/estimator/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/export/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/export/export_lib.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/feature_column/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/feature_column/base_feature_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/feature_column/dense_features.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/feature_column/dense_features_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/feature_column/sequence_feature_column.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/initializers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/initializers/initializers.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/initializers/initializers_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/elu.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/leaky_relu.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/prelu.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/relu.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/softmax.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/activation/thresholded_relu.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/attention/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/attention/additive_attention.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/attention/attention.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/attention/base_dense_attention.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/attention/multi_head_attention.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/base_conv.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/base_depthwise_conv.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/base_separable_conv.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv1d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv2d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/conv3d_transpose.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/depthwise_conv1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/depthwise_conv2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/separable_conv1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/convolutional/separable_conv2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/activation.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/dense.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/einsum_dense.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/embedding.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/identity.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/lambda_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/masking.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/core/tf_op_layer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/experimental/dynamic_embedding.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/experimental/dynamic_lookup.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/kernelized.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/locally_connected/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/locally_connected/locally_connected1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/locally_connected/locally_connected2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/locally_connected/locally_connected_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/add.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/average.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/base_merge.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/concatenate.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/dot.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/maximum.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/minimum.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/multiply.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/merging/subtract.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/noise.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/batch_normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/batch_normalization_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/group_normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/layer_normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/spectral_normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/normalization/unit_normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/average_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/average_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/average_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_global_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_global_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_global_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/base_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_average_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_average_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_average_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_max_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_max_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/global_max_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/max_pooling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/max_pooling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/pooling/max_pooling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/benchmarks/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/benchmarks/feature_column_benchmark.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/category_encoding.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/discretization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/hashed_crossing.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/hashing.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/image_preprocessing.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/index_lookup.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/integer_lookup.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/preprocessing_stage.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/preprocessing_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/preprocessing_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/string_lookup.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/preprocessing/text_vectorization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/activity_regularization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/alpha_dropout.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/dropout.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/gaussian_dropout.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/gaussian_noise.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/spatial_dropout1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/spatial_dropout2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/regularization/spatial_dropout3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/cropping1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/cropping2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/cropping3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/flatten.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/permute.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/repeat_vector.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/reshape.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/up_sampling1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/up_sampling2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/up_sampling3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/zero_padding1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/zero_padding2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/reshaping/zero_padding3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/abstract_rnn_cell.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/base_conv_lstm.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/base_conv_rnn.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/base_cudnn_rnn.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/base_rnn.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/base_wrapper.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/bidirectional.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/cell_wrappers.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/conv_lstm1d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/conv_lstm2d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/conv_lstm3d.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/cudnn_gru.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/cudnn_lstm.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/dropout_rnn_cell_mixin.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/gru.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/gru_lstm_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/gru_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/legacy_cell_wrappers.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/legacy_cells.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/lstm.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/lstm_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/rnn_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/simple_rnn.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/stacked_rnn_cells.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/rnn/time_distributed.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/layers/serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/base.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/convolutional.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/core.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/migration_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/normalization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/pooling.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/legacy_tf_layers/variable_scope_shim.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/losses.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/accuracy_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/base_metric.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/confusion_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/f_score_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/hinge_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/iou_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/probabilistic_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/py_metric.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/metrics/regression_metrics.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/autocast_variable.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/device_compatibility_check.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/loss_scale_optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/policy.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/mixed_precision/test_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/models/cloning.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/models/sharpness_aware_minimization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adadelta.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adafactor.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adagrad.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adam.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adamax.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/adamw.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/ftrl.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/adadelta.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/adagrad.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/adam.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/adamax.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/ftrl.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/gradient_descent.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/nadam.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/optimizer_v2.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy/rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/legacy_learning_rate_decay.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/lion.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/nadam.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/optimizer.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/optimizer_v1.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/rmsprop.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/schedules/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/schedules/learning_rate_schedule.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/sgd.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/optimizers/utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/premade_models/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/premade_models/linear.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/premade_models/wide_deep.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/preprocessing/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/preprocessing/image.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/preprocessing/sequence.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/preprocessing/text.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/regularizers.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/hdf5_format.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/model_config.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/save.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/base_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/constants.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/json_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/layer_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/load.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/metric_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/model_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/network_serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/order_preserving_set.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/save.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/save_impl.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/serialized_attributes.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saved_model/utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/saving_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/legacy/serialization.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/object_registration.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/pickle_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/saving_api.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/saving_lib.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/saving/serialization_lib.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/testing_infra/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/testing_infra/keras_doctest_lib.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/testing_infra/test_combinations.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/testing_infra/test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/tests/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/tests/get_config_samples.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/tests/model_architectures.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/tests/model_subclassing_test_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/audio_dataset.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/control_flow_util.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/conv_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/data_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/dataset_creator.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/dataset_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/feature_space.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/generic_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/image_dataset.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/image_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/io_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/keras_logging.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/kernelized_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/kpl_test_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/layer_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/losses_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/metrics_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/mode_keys.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/np_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/object_identity.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/sidecar_evaluator.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/steps_per_execution_tuning.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/text_dataset.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/tf_contextlib.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/tf_inspect.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/tf_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/timed_threads.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/timeseries_dataset.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/traceback_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/version_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/src/utils/vis_utils.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/utils/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/utils/experimental/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tf_keras/utils/legacy/__init__.py
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl-3.6.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl-3.6.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl-3.6.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl-3.6.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl-3.6.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/threadpoolctl.py
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/licenses/LICENCE
?? .venv_tf/lib/python3.12/site-packages/tqdm-4.67.3.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/tqdm/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/__main__.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_main.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_monitor.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_tqdm.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_tqdm_gui.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_tqdm_notebook.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_tqdm_pandas.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/_utils.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/asyncio.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/auto.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/autonotebook.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/cli.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/completion.sh
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/bells.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/concurrent.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/discord.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/itertools.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/logging.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/slack.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/telegram.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/contrib/utils_worker.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/dask.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/gui.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/keras.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/notebook.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/rich.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/std.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/tk.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/tqdm.1
?? .venv_tf/lib/python3.12/site-packages/tqdm/utils.py
?? .venv_tf/lib/python3.12/site-packages/tqdm/version.py
?? .venv_tf/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/typing_extensions.py
?? .venv_tf/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/licenses/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/urllib3/__init__.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/_base_connection.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/_collections.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/_request_methods.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/_version.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/connection.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/connectionpool.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/__init__.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/__init__.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/connection.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/fetch.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/request.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/emscripten/response.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/pyopenssl.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/contrib/socks.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/exceptions.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/fields.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/filepost.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/http2/__init__.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/http2/connection.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/http2/probe.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/poolmanager.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/py.typed
?? .venv_tf/lib/python3.12/site-packages/urllib3/response.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/__init__.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/connection.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/proxy.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/request.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/response.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/retry.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/ssl_.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/ssl_match_hostname.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/ssltransport.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/timeout.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/url.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/util.py
?? .venv_tf/lib/python3.12/site-packages/urllib3/util/wait.py
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/entry_points.txt
?? .venv_tf/lib/python3.12/site-packages/wheel-0.47.0.dist-info/licenses/LICENSE.txt
?? .venv_tf/lib/python3.12/site-packages/wheel/__init__.py
?? .venv_tf/lib/python3.12/site-packages/wheel/__main__.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_bdist_wheel.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/__init__.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/convert.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/info.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/pack.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/tags.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_commands/unpack.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_metadata.py
?? .venv_tf/lib/python3.12/site-packages/wheel/_setuptools_logging.py
?? .venv_tf/lib/python3.12/site-packages/wheel/bdist_wheel.py
?? .venv_tf/lib/python3.12/site-packages/wheel/macosx_libfile.py
?? .venv_tf/lib/python3.12/site-packages/wheel/metadata.py
?? .venv_tf/lib/python3.12/site-packages/wheel/wheelfile.py
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/INSTALLER
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/METADATA
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/RECORD
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/WHEEL
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/licenses/LICENSE
?? .venv_tf/lib/python3.12/site-packages/wrapt-2.2.1.dist-info/top_level.txt
?? .venv_tf/lib/python3.12/site-packages/wrapt-stubs/__init__.pyi
?? .venv_tf/lib/python3.12/site-packages/wrapt/__init__.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/__wrapt__.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/_wrappers.c
?? .venv_tf/lib/python3.12/site-packages/wrapt/_wrappers.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/wrapt/arguments.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/caching.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/decorators.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/importer.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/patches.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/proxies.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/signature.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/synchronization.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/weakrefs.py
?? .venv_tf/lib/python3.12/site-packages/wrapt/wrappers.py
?? .venv_tf/lib/python3.12/site-packages/yaml/__init__.py
?? .venv_tf/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so
?? .venv_tf/lib/python3.12/site-packages/yaml/composer.py
?? .venv_tf/lib/python3.12/site-packages/yaml/constructor.py
?? .venv_tf/lib/python3.12/site-packages/yaml/cyaml.py
?? .venv_tf/lib/python3.12/site-packages/yaml/dumper.py
?? .venv_tf/lib/python3.12/site-packages/yaml/emitter.py
?? .venv_tf/lib/python3.12/site-packages/yaml/error.py
?? .venv_tf/lib/python3.12/site-packages/yaml/events.py
?? .venv_tf/lib/python3.12/site-packages/yaml/loader.py
?? .venv_tf/lib/python3.12/site-packages/yaml/nodes.py
?? .venv_tf/lib/python3.12/site-packages/yaml/parser.py
?? .venv_tf/lib/python3.12/site-packages/yaml/reader.py
?? .venv_tf/lib/python3.12/site-packages/yaml/representer.py
?? .venv_tf/lib/python3.12/site-packages/yaml/resolver.py
?? .venv_tf/lib/python3.12/site-packages/yaml/scanner.py
?? .venv_tf/lib/python3.12/site-packages/yaml/serializer.py
?? .venv_tf/lib/python3.12/site-packages/yaml/tokens.py
?? .venv_tf/lib64
?? .venv_tf/pyvenv.cfg
?? =0.45.0
?? AGENTS.md
?? "GEMINI - \353\263\265\354\202\254\353\263\270.md:Zone.Identifier"
?? GEMINI.md
?? MDIQA/
?? "TROUBLESHOOTING - \353\263\265\354\202\254\353\263\270.md:Zone.Identifier"
?? analyze_datasets.py
?? android_gemma_latency_harness/README.md
?? android_gemma_latency_harness/app/build.gradle
?? android_gemma_latency_harness/app/src/main/AndroidManifest.xml
?? android_gemma_latency_harness/app/src/main/java/com/example/gemmaharness/BenchmarkRunner.kt
?? android_gemma_latency_harness/app/src/main/java/com/example/gemmaharness/JsonValidator.kt
?? android_gemma_latency_harness/app/src/main/java/com/example/gemmaharness/MainActivity.kt
?? android_gemma_latency_harness/app/src/main/res/layout/activity_main.xml
?? android_gemma_latency_harness/app/src/main/res/values/strings.xml
?? android_gemma_latency_harness/build.gradle
?? android_gemma_latency_harness/settings.gradle
?? check_severe.py
?? checklist.md
?? checkpoint_drive_listing.csv
?? checkpoint_local_inventory.csv
?? checkpoint_mapping_candidates.csv
?? configs/aesthetic_weight_lab.yaml
?? configs/aesthetic_weight_lab_alamp_finetuned.yaml
?? configs/gemma4_e4b_aesthetic3_dataset_smoke.json
?? configs/gemma4_e4b_aesthetic3_dataset_smoke_pass2.json
?? configs/gemma4_e4b_aesthetic3_teacher.json
?? configs/gemma4_e4b_aesthetic3_teacher_overrides_pass2.json
?? configs/gemma4_e4b_aesthetic3_train_smoke.json
?? configs/gemma4_e4b_aesthetic3_train_smoke_pass2.json
?? configs/gemma4_e4b_aesthetic3_train_smoke_pass2b.json
?? configs/gemma4_e4b_gemini_teacher_282_train.json
?? configs/paper_benchmarks/alamp_paper_ava_v2_graph_lite.yaml
?? configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r2.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r4.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r8.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/agg_max.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/dilation_paper_3_6_12_18.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/graph_blocks_2.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/input_300.yaml
?? configs/paper_benchmarks/rgnet_v1_ablations/raw_lse_then_sigmoid.yaml
?? configs/paper_benchmarks/rgnet_v1_full_candidates/agg_mean_full.yaml
?? configs/paper_benchmarks/rgnet_v1_full_candidates/combined_paper_dilation_mean.yaml
?? configs/paper_benchmarks/rgnet_v1_full_candidates/dilation_paper_3_6_12_18_full.yaml
?? configs/stage5_student_server.json
?? configs/stage5_student_server_weights.json
?? configs/student_distill_baseline.json
?? configs/student_distill_conservative_full_aadb_20260423.json
?? configs/student_distill_dry_run.json
?? configs/student_distill_full_aadb.json
?? configs/student_distill_full_run.json
?? configs/student_distill_teacher.json
?? configs/student_distill_teacher_conservative_aadb.json
?? context-notes.md
?? data/TW/1780467784142.jpg
?? data/TW/1780467785285.jpg
?? data/TW/IMG_2642.JPG
?? data/Test_weight/Bad/1626262173415.jpg
?? data/Test_weight/Bad/1739435940264-10.jpg
?? data/Test_weight/Bad/1739435940264-24.jpg
?? data/Test_weight/Bad/1739435965264-23.jpg
?? data/Test_weight/Bad/1777520144073.jpg
?? data/Test_weight/Bad/1777520144275.jpg
?? data/Test_weight/Bad/20210106_160007.jpg
?? data/Test_weight/Bad/20210713_125337.jpg
?? data/Test_weight/Bad/20210713_140803.jpg
?? data/Test_weight/Bad/20210713_141236.jpg
?? data/Test_weight/Bad/20210713_161741.jpg
?? data/Test_weight/Bad/20210721_160920.jpg
?? data/Test_weight/Bad/20210731_150004.jpg
?? data/Test_weight/Bad/IMG_2785.JPG
?? data/Test_weight/Good/1704296944756.jpg
?? data/Test_weight/Good/1720499666985.jpg
?? data/Test_weight/Good/1738350253185-8.jpg
?? data/Test_weight/Good/1739435968970-20.jpg
?? data/Test_weight/Good/1739436180999-6.jpg
?? data/Test_weight/Good/1780467785285.jpg
?? data/Test_weight/Good/20231231_150055.jpg
?? data/Test_weight/Good/20240515_092724.jpg
?? data/Test_weight/Good/20250204_133404.jpg
?? data/Test_weight/Good/20250204_135520.jpg
?? data/Test_weight/Good/20250204_163550.jpg
?? data/Test_weight/Good/20250204_170733.jpg
?? data/Test_weight/Good/20250204_174237(0).jpg
?? data/Test_weight/Good/20250217_144845.jpg
?? data/Test_weight/Good/IMG_20240519_215827_510.jpg
?? data/Test_weight/Soso/1675342165226-5.jpg
?? data/Test_weight/Soso/1720499571122.jpg
?? data/Test_weight/Soso/1720499621231.jpg
?? data/Test_weight/Soso/1720499621878.jpg
?? data/Test_weight/Soso/1738350393995-14.jpg
?? data/Test_weight/Soso/1739435940264-15.jpg
?? data/Test_weight/Soso/1739656718436-7.jpg
?? data/Test_weight/Soso/1780467784142.jpg
?? data/Test_weight/Soso/20210818_112220.jpg
?? data/Test_weight/Soso/IMG_2607.JPG
?? data/Test_weight/Soso/IMG_2642.JPG
?? data/debug/acut_aesthetic_compare_device_log_20260603.txt
?? data/debug/c6_parity_20260527/android_c6_scores.csv
?? data/debug/c6_parity_20260527/android_topiq_tensor_fingerprint_rows.csv
?? debug_savedmodel_nan.py
?? diagnose_parity.py
?? docs/capstone_artifact_map_20260605.md
?? docs/color_aesthetics_kickoff_audit.md
?? docs/full_repository_directory_audit_20260605.md
?? docs/github_upload_audit_20260605.md
?? docs/icaa_official_repo_dataset_audit.md
?? docs/pozy_app_stage5_aesthetic_student_handoff.md
?? docs/repository_inventory_20260605.md
?? docs/student_distillation_baseline.md
?? ds_decile.py
?? experiments/icaa_export_safe/export_safe_models/SoftHistogram.py
?? experiments/icaa_export_safe/export_safe_models/__init__.py
?? experiments/icaa_export_safe/export_safe_models/dat.py
?? experiments/icaa_export_safe/export_safe_models/dat_blocks.py
?? experiments/icaa_export_safe/reports/export_safe_review.md
?? experiments/icaa_export_safe/reports/flex_tflite_litert_feasibility_result.md
?? experiments/icaa_export_safe/reports/onnx_parity_debug_report.md
?? experiments/icaa_export_safe/reports/risky_ops_audit.md
?? experiments/icaa_export_safe/reports/tflite_conversion_feasibility_report.md
?? experiments/icaa_export_safe/scripts/evaluate_export_safe_full.py
?? experiments/icaa_export_safe/scripts/export_and_parity_onnx_export_safe.py
?? experiments/icaa_export_safe/scripts/export_and_parity_torchscript_export_safe.py
?? experiments/icaa_export_safe/scripts/inspect_onnx_graph.py
?? experiments/icaa_export_safe/scripts/list_tflite_ops.py
?? experiments/icaa_export_safe/scripts/litert_smoke_test.py
?? experiments/icaa_export_safe/scripts/parity_original_vs_export_safe.py
?? experiments/icaa_export_safe/scripts/tflite_smoke_test.py
?? experiments/icaa_export_safe_v2/export_safe_models/SoftHistogram.py
?? experiments/icaa_export_safe_v2/export_safe_models/__init__.py
?? experiments/icaa_export_safe_v2/export_safe_models/dat.py
?? experiments/icaa_export_safe_v2/export_safe_models/dat_blocks.py
?? experiments/icaa_export_safe_v2/reports/grid_sample_free_implementation_result.md
?? experiments/icaa_export_safe_v2/reports/grid_sample_free_plan.md
?? experiments/icaa_export_safe_v2/scripts/export_and_parity_onnx_v2.py
?? experiments/icaa_export_safe_v2/scripts/parity_tflite_fp32_v2.py
?? experiments/icaa_export_safe_v2/scripts/parity_v1_vs_v2_pytorch.py
?? experiments/icaa_export_safe_v3/export_safe_models/SoftHistogram.py
?? experiments/icaa_export_safe_v3/export_safe_models/__init__.py
?? experiments/icaa_export_safe_v3/export_safe_models/dat.py
?? experiments/icaa_export_safe_v3/export_safe_models/dat_blocks.py
?? experiments/icaa_export_safe_v3/reports/final_pooling_safe_result.md
?? experiments/icaa_export_safe_v3/scripts/convert_and_probe_tflite_v3.py
?? experiments/icaa_export_safe_v3/scripts/export_and_parity_onnx_v3.py
?? experiments/icaa_export_safe_v3/scripts/parity_v2_vs_v3_pytorch.py
?? experiments/icaa_tf_native/reports/stage_a_plan.md
?? experiments/icaa_tf_native/reports/stage_a_state_dict_inspection.md
?? experiments/icaa_tf_native/reports/stage_b_basic_layer_parity.md
?? experiments/icaa_tf_native/reports/stage_b_plan.md
?? experiments/icaa_tf_native/reports/stage_c_patch_projection_parity.md
?? experiments/icaa_tf_native/reports/stage_c_plan.md
?? experiments/icaa_tf_native/reports/stage_d_plan.md
?? experiments/icaa_tf_native/reports/stage_d_soft_histogram_parity.md
?? experiments/icaa_tf_native/reports/stage_e_local_attention_block_parity.md
?? experiments/icaa_tf_native/reports/stage_e_plan.md
?? experiments/icaa_tf_native/reports/stage_f_deformable_attention_parity.md
?? experiments/icaa_tf_native/reports/stage_f_plan.md
?? experiments/icaa_tf_native/reports/stage_i_savedmodel_export.md
?? experiments/icaa_tf_native/reports/tf_native_dat_reimplementation_plan.md
?? experiments/icaa_tf_native/scripts/inspect_pytorch_state_dict.py
?? experiments/icaa_tf_native/scripts/stage_b_basic_layer_parity.py
?? experiments/icaa_tf_native/scripts/stage_c_patch_projection_parity.py
?? experiments/icaa_tf_native/scripts/stage_d_soft_histogram_parity.py
?? experiments/icaa_tf_native/scripts/stage_e_local_attention_block_parity.py
?? experiments/icaa_tf_native/scripts/stage_f_deformable_attention_parity.py
?? experiments/icaa_tf_native/tf_models/patch_projection.py
?? experiments/icaa_tf_native/tf_models/soft_histogram.py
?? experiments/icaa_tf_native/tf_models/utils.py
?? experiments/icaa_tf_native/weight_maps/pytorch_state_dict_inventory.csv
?? experiments/icaa_tf_native/weight_maps/pytorch_state_dict_summary.json
?? exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.comparison.json
?? exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.metadata.json
?? exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.verify.json
?? exports/tflite/student_conservative_aesthetic_full_aadb_20260423.metadata.json
?? external/A-Lamp_external_audit/A-Lamp/
?? external/icaa_official_repo/
?? finalize_csvs.py
?? forWeights/
?? forWeights2/models/aesthetic/nima_mobile_fixed_preproc.metadata.json
?? forWeights2/models/aesthetic/rgnet_pil_resize_aadb.metadata.json
?? functions/node_modules/@grpc/grpc-js/build/src/generated/xds/data/orca/v3/OrcaLoadReport.d.ts
?? functions/node_modules/@grpc/grpc-js/build/src/generated/xds/data/orca/v3/OrcaLoadReport.js
?? functions/node_modules/@grpc/grpc-js/build/src/generated/xds/data/orca/v3/OrcaLoadReport.js.map
?? functions/node_modules/@grpc/grpc-js/proto/xds/xds/data/orca/v3/orca_load_report.proto
?? functions/node_modules/@grpc/grpc-js/src/generated/xds/data/orca/v3/OrcaLoadReport.ts
?? functions/node_modules/@protobufjs/inquire/tests/data/array.js
?? functions/node_modules/@protobufjs/inquire/tests/data/emptyArray.js
?? functions/node_modules/@protobufjs/inquire/tests/data/emptyObject.js
?? functions/node_modules/@protobufjs/inquire/tests/data/object.js
?? functions/node_modules/@protobufjs/utf8/tests/data/utf8.txt
?? generate_preprocessed_previews.py
?? gpu_check.py
?? inspect_sm_weights.py
?? missing_checkpoint_actions.csv
?? mixed112_files.txt
?? onnx2tf_full_help.txt
?? onnx2tf_help.txt
?? open_clip/
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/file_inventory.csv
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/join_key_audit.csv
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/leakage_audit.md
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/metric_reproduction_summary.csv
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/recommended_next_action.md
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/report.md
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/row_count_summary.csv
?? outputs/acut_aesthetic_benchmark_csv_audit_20260603/score_distribution_summary.csv
?? outputs/acut_aesthetic_weight_multiobjective_20260603/acut_user_rank_input_template.csv
?? outputs/acut_aesthetic_weight_multiobjective_20260603/multiobjective_weight_results.csv
?? outputs/acut_aesthetic_weight_multiobjective_20260603/report.md
?? outputs/acut_aesthetic_weight_multiobjective_20260603/top_candidates.csv
?? outputs/acut_capstone_final_report_20260526/evidence_index.md
?? outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv
?? outputs/acut_capstone_final_report_20260526/report.md
?? outputs/acut_capstone_final_report_20260526/report_generation_log.md
?? outputs/acut_final_ratio_core_eval_20260603/device_log_selected_ratio_distribution.csv
?? outputs/acut_final_ratio_core_eval_20260603/optimal_ratios.csv
?? outputs/acut_final_ratio_core_eval_20260603/product_gold_selected_ratio_metrics.csv
?? outputs/acut_final_ratio_core_eval_20260603/report.md
?? outputs/acut_final_ratio_core_eval_20260603/selected_ratio_ava_aadb_metrics.csv
?? outputs/acut_final_ratio_core_eval_20260603/selected_ratio_summary.json
?? outputs/acut_model_core_code_audit_20260601/core_code_map.csv
?? outputs/acut_model_core_code_audit_20260601/model_usage_matrix.csv
?? outputs/acut_model_core_code_audit_20260601/report.md
?? outputs/acut_model_core_code_audit_20260601/report_generation_log.md
?? outputs/acut_model_core_code_audit_20260601/risks_and_mismatches.md
?? outputs/acut_model_core_training_code_excerpt_20260601/code_excerpt_index.csv
?? outputs/acut_model_core_training_code_excerpt_20260601/presentation_code_appendix.md
?? outputs/acut_model_core_training_code_excerpt_20260601/report.md
?? outputs/acut_model_core_training_code_excerpt_20260601/report_generation_log.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/common_export_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/flive_core_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/icaa_core_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/koniq_core_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/mobile_alamp_v2_core_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/nima_core_code.md
?? outputs/acut_model_core_training_code_excerpt_20260601/snippets/rgnet_core_code.md
?? outputs/acut_model_training_code_audit_20260601/evidence_index.csv
?? outputs/acut_model_training_code_audit_20260601/missing_training_provenance.md
?? outputs/acut_model_training_code_audit_20260601/report.md
?? outputs/acut_model_training_code_audit_20260601/report_generation_log.md
?? outputs/acut_model_training_code_audit_20260601/training_code_map.csv
?? outputs/acut_product_gold_weight_eval_20260603/class_mean_scores.csv
?? outputs/acut_product_gold_weight_eval_20260603/failures.csv
?? outputs/acut_product_gold_weight_eval_20260603/fixed_candidate_metrics.csv
?? outputs/acut_product_gold_weight_eval_20260603/product_gold_manifest.csv
?? outputs/acut_product_gold_weight_eval_20260603/product_gold_predictions.csv
?? outputs/acut_product_gold_weight_eval_20260603/report.md
?? outputs/acut_product_gold_weight_eval_20260603/top_weight_candidates.csv
?? outputs/acut_product_gold_weight_eval_20260603/weight_search_results.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/class_mean_scores.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/domain_caveat_analysis.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/failures.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/fixed_candidate_metrics.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/pooled_weight_search_results.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/report.md
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/test_vila_manifest.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/test_vila_predictions.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/top_weight_candidates.csv
?? outputs/acut_product_gold_weight_eval_test_vila_20260603/weight_search_results.csv
?? outputs/acut_weekly_research_journal_20260526/experiment_timeline.csv
?? outputs/acut_weekly_research_journal_20260526/report_generation_log.md
?? outputs/acut_weekly_research_journal_20260526/weekly_evidence_index.md
?? outputs/acut_weekly_research_journal_20260526/weekly_research_journal.md
?? outputs/acut_weekly_research_journal_20260526/weekly_summary.csv
?? outputs/acut_weekly_research_journal_v2_20260526/README.md
?? outputs/acut_weekly_research_journal_v2_20260526/evidence_index.csv
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/file_dates_mod.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/file_inventory.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_aesthetic.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_deployment.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_firebase.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_gemma_litert.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_technical.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/files_vlm_reasoning.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/potential_evidence_files.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_aesthetic.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_deployment.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_firebase.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_gemma_litert.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_technical.txt
?? outputs/acut_weekly_research_journal_v2_20260526/parsed/search_vlm_reasoning.txt
?? outputs/acut_weekly_research_journal_v2_20260526/report_generation_log.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/aesthetic_models.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/firebase_backend_integration.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/flutter_mobile_deployment.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/gemma_on_device_llm.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/technical_quality_models.md
?? outputs/acut_weekly_research_journal_v2_20260526/topics/vila_and_reasoning.md
?? outputs/acut_weekly_research_journal_v2_20260526/weekly_research_journal_combined.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_01_20260330_20260405.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_02_20260406_20260412.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_03_20260413_20260419.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_04_20260420_20260426.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_05_20260427_20260503.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_06_20260504_20260510.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_07_20260511_20260517.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_08_20260518_20260524.md
?? outputs/acut_weekly_research_journal_v2_20260526/weeks/week_09_20260525_20260531.md
?? outputs/aesthetic_ensemble_production_precheck_20260603/codex_implementation_handoff.md
?? outputs/aesthetic_ensemble_production_precheck_20260603/mac_checksum_commands.md
?? outputs/aesthetic_ensemble_production_precheck_20260603/model_artifact_inventory.csv
?? outputs/aesthetic_ensemble_production_precheck_20260603/parity_test_plan.md
?? outputs/aesthetic_ensemble_production_precheck_20260603/production_precheck_report.md
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/baseline_single_model_metrics.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/bootstrap_summary.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/checklist.md
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/context-notes.md
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/cv5_summary.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/final_recommendation.json
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/report.md
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/scale_verification.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/selected_weight_test_metrics.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/top_weights_group_a.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/top_weights_group_b_icaa_cap_015.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/top_weights_group_b_icaa_cap_020.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/top_weights_group_b_icaa_cap_025.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/top_weights_group_b_unconstrained.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603/weight_search_results_all.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603_audit/aesthetic_ensemble_config.json
?? outputs/aesthetic_ensemble_weight_search_complete_20260603_audit/audit_summary.csv
?? outputs/aesthetic_ensemble_weight_search_complete_20260603_audit/weight_search_audit.md
?? outputs/aesthetic_full_paper_benchmark_gap_analysis_20260603_audit/corrected_metrics_summary.csv
?? outputs/aesthetic_full_paper_benchmark_gap_analysis_20260603_audit/full_paper_benchmark_gap_analysis.md
?? outputs/aesthetic_full_paper_benchmark_gap_analysis_20260603_audit/remaining_actions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/checklist.md
?? outputs/aesthetic_mobile_paper_benchmark_20260603/context-notes.md
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/alamp_ava/alamp_v2_failures.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/alamp_ava/alamp_v2_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/alamp_ava/alamp_v2_metrics_fixed.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/alamp_ava/alamp_v2_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_spaq/icaa_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_spaq/icaa_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_failures.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_metrics_fixed.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/plan.md
?? outputs/aesthetic_mobile_paper_benchmark_20260603/report.md
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/alamp_ava/alamp_v2_failures.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/alamp_ava/alamp_v2_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/alamp_ava/alamp_v2_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/icaa_icaa17k/icaa_failures.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/icaa_icaa17k/icaa_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/icaa_icaa17k/icaa_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/nima_ava/nima_failures.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/nima_ava/nima_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/nima_ava/nima_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/rgnet_aadb/rgnet_metrics.json
?? outputs/aesthetic_mobile_paper_benchmark_20260603/smoke/rgnet_aadb/rgnet_predictions.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603_audit/audit_stats.csv
?? outputs/aesthetic_mobile_paper_benchmark_20260603_audit/benchmark_validity_audit.md
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/alamp_adaptive_patch_benchmark_report.md
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/alamp_adaptive_patch_feasibility.md
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/diagnostic_1000/alamp_adaptive_metrics.json
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/diagnostic_1000/alamp_adaptive_predictions.csv
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/final_claims_for_capstone.md
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_failures.csv
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_metrics.json
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/rgnet_aadb_split_evidence.md
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/rgnet_recomputed_metrics.json
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/smoke_adaptive/alamp_adaptive_metrics.json
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/smoke_adaptive/alamp_adaptive_predictions.csv
?? outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/summary.csv
?? outputs/alamp_gpu_acceleration_audit_20260603/report.md
?? outputs/c6_android_wsl_parity_20260527/android_vs_wsl_parity.csv
?? outputs/c6_android_wsl_parity_20260527/report.md
?? outputs/c6_android_wsl_parity_20260527/wsl_c6_scores.csv
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/android_vs_wsl_parity.csv
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/checklist.md
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/context-notes.md
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/plan.md
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/report.md
?? outputs/c6_android_wsl_parity_exif_fixed_20260529/wsl_c6_scores.csv
?? outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/mismatch_attribution.csv
?? outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/preprocess_debug/preprocess_stats.csv
?? outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/report.md
?? outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/top_mismatch_image_metadata.csv
?? outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/wsl_with_orientation/parity_with_orientation.csv
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/backend_summary.csv
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/checklist.md
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/context-notes.md
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/per_image_comparison_best.csv
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/plan.md
?? outputs/c6_topiq_preprocess_backend_ablation_20260529/report.md
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/backend_fingerprint_summary.csv
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/checklist.md
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/context-notes.md
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/per_image_fingerprint_comparison.csv
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/plan.md
?? outputs/c6_topiq_tensor_fingerprint_compare_20260529/report.md
?? outputs/capstone_model_audit_20260711/00_inventory/checklist.md
?? outputs/capstone_model_audit_20260711/00_inventory/commands_executed.txt
?? outputs/capstone_model_audit_20260711/00_inventory/context-notes.md
?? outputs/capstone_model_audit_20260711/00_inventory/environment_snapshot.txt
?? outputs/dryrun_rgnet_ava_repro_full_1ep_20260526/train_summary.json
?? outputs/dryrun_rgnet_ava_repro_full_1ep_20260526/training_history.csv
?? outputs/final_c6_log_only_readiness_20260528/readiness_summary.csv
?? outputs/final_c6_log_only_readiness_20260528/report.md
?? outputs/final_c6_topiq_koniq_technical_score_decision_20260527/decision_summary.csv
?? outputs/final_c6_topiq_koniq_technical_score_decision_20260527/report.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/aesthetic_iaa_assessment.csv
?? outputs/full_iqa_iaa_model_design_audit_20260611/codex_final_verification.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/ensemble_design_review.csv
?? outputs/full_iqa_iaa_model_design_audit_20260611/full_model_design_audit.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/mac_submitted_app_supplemental_verification.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/mobile_deployment_risk_review.csv
?? outputs/full_iqa_iaa_model_design_audit_20260611/next_actions.csv
?? outputs/full_iqa_iaa_model_design_audit_20260611/production_recommendation.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/revision_changelog.md
?? outputs/full_iqa_iaa_model_design_audit_20260611/technical_iqa_assessment.csv
?? outputs/full_iqa_iaa_model_design_audit_20260611/thesis_writing_notes.md
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibrated_2k_manifest.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_failure_cases.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_parameters.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_policy_comparison.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_policy_metrics.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_range_audit.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_safety_cases.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/next_actions.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/recommended_adjusted_policy.json
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603/report.md
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibrated_2k_manifest.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_failure_cases.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_parameters.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_policy_comparison.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_policy_metrics.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_range_audit.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_safety_cases.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/next_actions.csv
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/recommended_adjusted_policy.json
?? outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/report.md
?? outputs/hybrid_teacher_adjusted_calibration_design_20260603/adjusted_calibration_policy_spec.md
?? outputs/hybrid_teacher_adjusted_calibration_design_20260603/previous_calibration_failure_analysis.md
?? outputs/hybrid_teacher_final_label_scope_decision_20260603/fallback_limitation_summary.csv
?? outputs/hybrid_teacher_final_label_scope_decision_20260603/known_dataset_label_scope.csv
?? outputs/hybrid_teacher_final_label_scope_decision_20260603/next_actions.csv
?? outputs/hybrid_teacher_final_label_scope_decision_20260603/report.md
?? outputs/hybrid_teacher_final_label_scope_decision_20260603/technical_aesthetic_mismatch_cases.csv
?? outputs/hybrid_teacher_label_design_review_20260603/hybrid_label_dryrun_manifest.csv
?? outputs/hybrid_teacher_label_design_review_20260603/hybrid_label_dryrun_validation.csv
?? outputs/hybrid_teacher_label_design_review_20260603/hybrid_label_safety_cases.csv
?? outputs/hybrid_teacher_label_design_review_20260603/hybrid_teacher_label_spec.md
?? outputs/hybrid_teacher_label_design_review_20260603/next_actions.csv
?? outputs/hybrid_teacher_label_design_review_20260603/report.md
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_distribution_summary_2k.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_validation.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_failure_cases_2k.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_policy_comparison_2k.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_safety_cases_2k.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/next_actions.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/range_calibration_audit.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/range_calibration_recommendations.csv
?? outputs/hybrid_teacher_label_design_review_2k_20260603/report.md
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_distribution_summary_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_dryrun_10k_manifest.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_dryrun_10k_validation.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_failure_cases_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_safety_cases_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/next_actions.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/policy_f_shift_cap3_summary_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/range_calibration_audit_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603/report.md
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_distribution_summary_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_dryrun_10k_manifest.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_dryrun_10k_validation.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_failure_cases_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_safety_cases_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/next_actions.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/policy_f_shift_cap3_summary_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/range_calibration_audit_10k.csv
?? outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/report.md
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibrated_2k_manifest.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_failure_cases.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_parameters.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_policy_comparison.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_policy_metrics.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_range_audit.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_safety_cases.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/next_actions.csv
?? outputs/hybrid_teacher_range_calibration_2k_20260603/recommended_policy.json
?? outputs/hybrid_teacher_range_calibration_2k_20260603/report.md
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/data_inventory_notes.md
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/report.md
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/test_weight_scores.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/test_weight_technical_scores.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_data_inventory.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_distribution_summary.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_dryrun_manifest.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_safety_cases.csv
?? outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_validation.csv
?? outputs/icaa_int8_qat_audit_20260601/artifact_inventory.csv
?? outputs/icaa_int8_qat_audit_20260601/benchmark_plan.md
?? outputs/icaa_int8_qat_audit_20260601/conversion_feasibility.md
?? outputs/icaa_int8_qat_audit_20260601/int8_ptq_plan.md
?? outputs/icaa_int8_qat_audit_20260601/next_action.md
?? outputs/icaa_int8_qat_audit_20260601/qat_plan.md
?? outputs/icaa_int8_qat_audit_20260601/report.md
?? outputs/known_dataset_full_label_coverage_audit_20260603/dataset_coverage_breakdown.csv
?? outputs/known_dataset_full_label_coverage_audit_20260603/koniq_missing_reason_summary.csv
?? outputs/known_dataset_full_label_coverage_audit_20260603/next_actions.csv
?? outputs/known_dataset_full_label_coverage_audit_20260603/recommended_scope_decision.json
?? outputs/known_dataset_full_label_coverage_audit_20260603/report.md
?? outputs/known_dataset_full_label_coverage_audit_20260603/spaq_scope_decision.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/all_variants_results.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/best_variant_android_vs_wsl.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/c6_variant_summary.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/koniq_variant_summary.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/report.md
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/top_c6_mismatches.csv
?? outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/top_koniq_mismatches.csv
?? outputs/mdiqa_checkpoint_source_audit_20260601/report.md
?? outputs/mdiqa_correct_protocol_full_benchmark_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_correct_protocol_full_benchmark_20260601/dataset_metrics.csv
?? outputs/mdiqa_correct_protocol_full_benchmark_20260601/filter_audit.csv
?? outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv
?? outputs/mdiqa_correct_protocol_full_benchmark_20260601/report.md
?? outputs/mdiqa_cuda_diagnosis_20260601/report.md
?? outputs/mdiqa_distillation_50k_full_targets_20260608/cache/huggingface/version.txt
?? outputs/mdiqa_distillation_50k_full_targets_20260608/dataset_metrics.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/filter_audit.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_cache_merge_audit.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_distribution_summary.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_failure_cases.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_leakage_audit.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/next_actions.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/remaining_mdiqa_predictions.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/remaining_missing_manifest.csv
?? outputs/mdiqa_distillation_50k_full_targets_20260608/report.md
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_policy_summary.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_preview.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/dataset_balance_summary.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/leakage_risk_audit.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/mdiqa_cache_inventory.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/missing_mdiqa_inference_estimate.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/next_actions.csv
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/recommendation.json
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/report.md
?? outputs/mdiqa_distillation_50k_manifest_audit_20260608/runtime_estimate.md
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/cache/huggingface/version.txt
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/dataset_metrics.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/filter_audit.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/next_actions.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/report.md
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_distribution_summary.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_failure_cases.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_leakage_audit.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_manifest.csv
?? outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_mdiqa_predictions.csv
?? outputs/mdiqa_flive_image_test_sanity_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_flive_image_test_sanity_20260601/flive_image_test_manifest.csv
?? outputs/mdiqa_flive_image_test_sanity_20260601/metrics.csv
?? outputs/mdiqa_flive_image_test_sanity_20260601/predictions.csv
?? outputs/mdiqa_flive_image_test_sanity_20260601/report.md
?? outputs/mdiqa_flive_image_test_sanity_20260601/sensitivity_checks.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/checkpoint_mapping_audit.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/flive_manifest_audit.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/flive_metric_sensitivity.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/next_actions.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/paper_flive_claims.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/protocol_comparison.csv
?? outputs/mdiqa_flive_protocol_audit_20260601/report.md
?? outputs/mdiqa_flive_protocol_audit_20260601/suspected_root_causes.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/current_cache_vs_official_scope.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/expansion_risk_matrix.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/missing_rows_estimate.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/next_actions.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/official_protocol_inventory.csv
?? outputs/mdiqa_official_protocol_expansion_audit_20260603/report.md
?? outputs/mdiqa_official_variant_sanity_20260601/best_variant_by_dataset.csv
?? outputs/mdiqa_official_variant_sanity_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_official_variant_sanity_20260601/checkpoint_inventory.csv
?? outputs/mdiqa_official_variant_sanity_20260601/dataset_metrics.csv
?? outputs/mdiqa_official_variant_sanity_20260601/failures.csv
?? outputs/mdiqa_official_variant_sanity_20260601/predictions.csv
?? outputs/mdiqa_official_variant_sanity_20260601/report.md
?? outputs/mdiqa_official_variant_sanity_20260601/variant_sanity_manifest.csv
?? outputs/mdiqa_sanity_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_sanity_20260601/dataset_metrics.csv
?? outputs/mdiqa_sanity_20260601/failures.csv
?? outputs/mdiqa_sanity_20260601/predictions.csv
?? outputs/mdiqa_sanity_20260601/report.md
?? outputs/mdiqa_sanity_20260601/sanity_manifest.csv
?? outputs/mdiqa_smoke_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_smoke_20260601/report.md
?? outputs/mdiqa_smoke_20260601/smoke_predictions.csv
?? outputs/mdiqa_student_final_no_go_report_20260611/decision_matrix.csv
?? outputs/mdiqa_student_final_no_go_report_20260611/next_actions.csv
?? outputs/mdiqa_student_final_no_go_report_20260611/report.md
?? outputs/mdiqa_student_final_no_go_report_20260611/research_takeaways.md
?? outputs/mdiqa_student_final_no_go_report_20260611/summary_table.csv
?? outputs/mdiqa_student_gap_analysis_20260608/gap_predictions_joined.csv
?? outputs/mdiqa_student_gap_analysis_20260608/gap_recommendation.json
?? outputs/mdiqa_student_gap_analysis_20260608/gap_summary_datasetwise.csv
?? outputs/mdiqa_student_gap_analysis_20260608/gap_summary_mos_bins.csv
?? outputs/mdiqa_student_gap_analysis_20260608/gap_summary_overall.csv
?? outputs/mdiqa_student_gap_analysis_20260608/gap_summary_student_vs_mdiqa_bins.csv
?? outputs/mdiqa_student_gap_analysis_20260608/next_actions.csv
?? outputs/mdiqa_student_gap_analysis_20260608/report.md
?? outputs/mdiqa_student_gap_analysis_20260608/safety_gap_cases.csv
?? outputs/mdiqa_student_gap_analysis_20260608/student_overpredict_low_mos_top30.csv
?? outputs/mdiqa_student_gap_analysis_20260608/student_underpredict_high_mos_top30.csv
?? outputs/mdiqa_student_gap_analysis_20260608/student_worse_top30.csv
?? outputs/mdiqa_student_gap_analysis_20260608/training_overlap_gap_summary.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/goal_alignment_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/mdiqa_code_correctness_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/mdiqa_variant_preprocess_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/next_actions.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/recommendation.json
?? outputs/mdiqa_student_pre_final_audit_20260609/report.md
?? outputs/mdiqa_student_pre_final_audit_20260609/split_leakage_distribution_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/student_model_architecture_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/student_preprocess_target_audit.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/target_outlier_cases.csv
?? outputs/mdiqa_student_pre_final_audit_20260609/training_data_quality_audit.csv
?? outputs/mdiqa_variant_sanity_20260601/best_variant_by_dataset.csv
?? outputs/mdiqa_variant_sanity_20260601/cache/huggingface/version.txt
?? outputs/mdiqa_variant_sanity_20260601/checkpoint_inventory.csv
?? outputs/mdiqa_variant_sanity_20260601/dataset_metrics.csv
?? outputs/mdiqa_variant_sanity_20260601/failures.csv
?? outputs/mdiqa_variant_sanity_20260601/predictions.csv
?? outputs/mdiqa_variant_sanity_20260601/report.md
?? outputs/mdiqa_variant_sanity_20260601/variant_sanity_manifest.csv
?? outputs/mdiqa_wsl_feasibility_audit_20260601/report.md
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/checkpoint_recommendation.json
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/datasetwise_checkpoint_metrics.csv
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/next_actions.csv
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/prediction_distribution.csv
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md
?? outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/safety_cases.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/best_checkpoint_summary.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/comparison_against_short.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/failure_cases.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/history.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/metrics.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/report.md
?? outputs/mobile_student_iqa_14105_full_controlled_20260604/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/best_checkpoint_summary.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/checklist.md
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/comparison_against_short.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/context-notes.md
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/failure_cases.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/history.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/metrics.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/report.md
?? outputs/mobile_student_iqa_14105_longer_controlled_20260604/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/checklist.md
?? outputs/mobile_student_iqa_14105_short_train_20260604/comparison_against_smoke.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/context-notes.md
?? outputs/mobile_student_iqa_14105_short_train_20260604/failure_cases.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/history.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/metrics.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_short_train_20260604/report.md
?? outputs/mobile_student_iqa_14105_short_train_20260604/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_smoke_20260603/checklist.md
?? outputs/mobile_student_iqa_14105_smoke_20260603/context-notes.md
?? outputs/mobile_student_iqa_14105_smoke_20260603/failure_cases.csv
?? outputs/mobile_student_iqa_14105_smoke_20260603/history.csv
?? outputs/mobile_student_iqa_14105_smoke_20260603/metrics.csv
?? outputs/mobile_student_iqa_14105_smoke_20260603/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_smoke_20260603/report.md
?? outputs/mobile_student_iqa_14105_smoke_20260603/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/failure_cases.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/history.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/metrics.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/report.md
?? outputs/mobile_student_iqa_14105_smoke_gpu_20260603/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/failure_cases.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/history.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/metrics.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/prediction_summary.csv
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/report.md
?? outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/checklist.md
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/context-notes.md
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/hybrid_runtime_design.md
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/model_size_summary.csv
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/next_actions.csv
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/report.md
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/sample_manifest_used.csv
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/tflite_model_analyzer.txt
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/tflite_parity_metrics.csv
?? outputs/mobile_student_iqa_14105_tflite_export_20260604/tflite_parity_predictions.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/failure_cases.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/history.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/metrics.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/predictions_test.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/predictions_train.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/predictions_val.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/safety_metrics.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/test_manifest.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/train_manifest.csv
?? outputs/mobile_student_iqa_50k_controlled_b0_20260608/val_manifest.csv
?? outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/gpu_preflight_commands.txt
?? outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/gpu_preflight_outputs.txt
?? outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/gpu_preflight_report.md
?? outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/b0_flive_worst_cases.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/b0_official_safety_cases.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/b0_overpredict_low_mos_top50.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/b0_underpredict_high_mos_top50.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/b0_worse_than_14k_top50.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/dataset_weighting_policy_summary.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/datasetwise_degradation_summary.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/internal_vs_official_shift_summary.json
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/official_paired_predictions.csv
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/recommendation.json
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md
?? outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/target_distribution_shift_summary.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/failure_cases.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/history.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/metrics.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/next_actions.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/predictions_test.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/predictions_train.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/predictions_val.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/report.md
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/safety_metrics.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/target_scaling_audit.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/test_manifest.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/train_manifest.csv
?? outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/val_manifest.csv
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/next_actions.csv
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/official_label_agreement_summary.csv
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/ranking_loss_design_summary.csv
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/recommended_phase21_plan.json
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/report.md
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/target_scaling_distribution_audit.csv
?? outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/target_scaling_policy_summary.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/checklist.md
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/context-notes.md
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/failure_cases.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/history.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/metrics.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/next_actions.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/predictions_test.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/predictions_train.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/predictions_val.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/ranking_pair_stats.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/report.md
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/safety_metrics.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/test_manifest.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/train_manifest.csv
?? outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/val_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/failure_cases.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/history.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/predictions_test.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/predictions_train.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/predictions_val.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/report.md
?? outputs/mobile_student_iqa_50k_smoke_20260608/test_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/train_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_20260608/val_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/failure_cases.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/history.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/report.md
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/test_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/train_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/val_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/failure_cases.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/history.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/metrics.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/next_actions.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/predictions_test.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/predictions_train.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/predictions_val.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/report.md
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/test_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/train_manifest.csv
?? outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/val_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/checklist.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/context-notes.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/failure_cases.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/history.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/next_actions.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/predictions_test.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/predictions_train.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/predictions_val.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/report.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/safety_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/test_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/train_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/val_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/checklist.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/context-notes.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/dataset_weight_effective_contribution.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/datasetwise_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/failure_cases.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/history.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/next_actions.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/prediction_summary.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/predictions_test.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/predictions_train.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/predictions_val.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/safety_metrics.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/test_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/train_manifest.csv
?? outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/val_manifest.csv
?? outputs/mobile_student_iqa_v1_20260601/dataset_metrics.csv
?? outputs/mobile_student_iqa_v1_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v1_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v1_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v1_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v1_20260601/report.md
?? outputs/mobile_student_iqa_v1_20260601/safety_metrics.csv
?? outputs/mobile_student_iqa_v1_20260601/training_history.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/dataset_metrics.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/load_errors.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/predictions_test.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/predictions_train.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/predictions_val.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/report.md
?? outputs/mobile_student_iqa_v1_20260601_smoke/safety_metrics.csv
?? outputs/mobile_student_iqa_v1_20260601_smoke/training_history.csv
?? outputs/mobile_student_iqa_v2_20260601/dataset_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v2_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v2_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v2_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v2_20260601/report.md
?? outputs/mobile_student_iqa_v2_20260601/safety_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601/training_history.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/dataset_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/load_errors.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/predictions_test.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/predictions_train.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/predictions_val.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/report.md
?? outputs/mobile_student_iqa_v2_20260601_fixed/safety_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601_fixed/training_history.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/dataset_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/load_errors.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/predictions_test.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/predictions_train.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/predictions_val.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/report.md
?? outputs/mobile_student_iqa_v2_20260601_smoke/safety_metrics.csv
?? outputs/mobile_student_iqa_v2_20260601_smoke/training_history.csv
?? outputs/mobile_student_iqa_v2_error_analysis_20260601/report.md
?? outputs/mobile_student_iqa_v2_training_loop_audit_20260601/report.md
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/dataset_metrics.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/report.md
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/safety_metrics.csv
?? outputs/mobile_student_iqa_v2_training_loop_verify_20260601/training_history.csv
?? outputs/mobile_student_iqa_v3_20260601/dataset_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v3_20260601/mos_reference_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601/overall_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v3_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v3_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v3_20260601/report.md
?? outputs/mobile_student_iqa_v3_20260601/safety_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v3_20260601/training_history.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/dataset_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/load_errors.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/mos_reference_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/overall_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/predictions_test.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/predictions_train.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/predictions_val.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/report.md
?? outputs/mobile_student_iqa_v3_20260601_smoke/safety_metrics.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v3_20260601_smoke/training_history.csv
?? outputs/mobile_student_iqa_v3_error_analysis_20260601/dataset_prediction_summary.csv
?? outputs/mobile_student_iqa_v3_error_analysis_20260601/decile_error_summary.csv
?? outputs/mobile_student_iqa_v3_error_analysis_20260601/report.md
?? outputs/mobile_student_iqa_v3_error_analysis_20260601/severe_fp_analysis.csv
?? outputs/mobile_student_iqa_v3_error_analysis_20260601/top_abs_errors.csv
?? outputs/mobile_student_iqa_v4a_20260601/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601/dataset_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v4a_20260601/mos_reference_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601/overall_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v4a_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v4a_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v4a_20260601/report.md
?? outputs/mobile_student_iqa_v4a_20260601/safety_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4a_20260601/training_history.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke/load_errors.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/dataset_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/load_errors.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/mos_reference_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/overall_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/predictions_test.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/predictions_train.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/predictions_val.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/report.md
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/safety_metrics.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/training_history.csv
?? outputs/mobile_student_iqa_v4b_20260601/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601/load_errors.csv
?? outputs/mobile_student_iqa_v4b_20260601/posthoc_calibration_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601/predictions_test.csv
?? outputs/mobile_student_iqa_v4b_20260601/predictions_test_calibrated.csv
?? outputs/mobile_student_iqa_v4b_20260601/predictions_train.csv
?? outputs/mobile_student_iqa_v4b_20260601/predictions_val.csv
?? outputs/mobile_student_iqa_v4b_20260601/report.md
?? outputs/mobile_student_iqa_v4b_20260601/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4b_20260601/training_history.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/load_errors.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/posthoc_calibration_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/predictions_test.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/predictions_test_calibrated.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/predictions_train.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/predictions_val.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/report.md
?? outputs/mobile_student_iqa_v4b_20260601_sanity/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4b_20260601_sanity/training_history.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/load_errors.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/posthoc_calibration_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/predictions_test.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/predictions_test_calibrated.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/predictions_train.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/predictions_val.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke/training_history.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/checkpoint_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/load_errors.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/posthoc_calibration_metrics.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/predictions_test.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/predictions_test_calibrated.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/predictions_train.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/predictions_val.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/report.md
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/severe_false_positive_cases.csv
?? outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/training_history.csv
?? outputs/musiq_kaggle_failure_diagnosis_20260601/failed_image_inspection.csv
?? outputs/musiq_kaggle_failure_diagnosis_20260601/fallback_test_results.csv
?? outputs/musiq_kaggle_failure_diagnosis_20260601/report.md
?? outputs/musiq_kaggle_sanity_20260601/dataset_metrics.csv
?? outputs/musiq_kaggle_sanity_20260601/failures.csv
?? outputs/musiq_kaggle_sanity_20260601/predictions.csv
?? outputs/musiq_kaggle_sanity_20260601/report.md
?? outputs/musiq_kaggle_sanity_20260601/sanity_manifest.csv
?? outputs/musiq_kaggle_sanity_fallback_20260601/dataset_metrics.csv
?? outputs/musiq_kaggle_sanity_fallback_20260601/failures.csv
?? outputs/musiq_kaggle_sanity_fallback_20260601/predictions.csv
?? outputs/musiq_kaggle_sanity_fallback_20260601/report.md
?? outputs/musiq_kaggle_sanity_fallback_20260601/sanity_manifest.csv
?? outputs/musiq_kaggle_smoke_20260601/report.md
?? outputs/musiq_kaggle_smoke_20260601/smoke_predictions.csv
?? outputs/musiq_wsl_kaggle_audit_20260601/musiq_candidate_variants.csv
?? outputs/musiq_wsl_kaggle_audit_20260601/musiq_local_artifacts.csv
?? outputs/musiq_wsl_kaggle_audit_20260601/report.md
?? outputs/official_protocol_benchmark_manifest_20260604/flive_official_image_test_manifest.csv
?? outputs/official_protocol_benchmark_manifest_20260604/flive_official_patch_test_manifest.csv
?? outputs/official_protocol_benchmark_manifest_20260604/internal_vs_official_overlap.csv
?? outputs/official_protocol_benchmark_manifest_20260604/koniq_official_test_manifest.csv
?? outputs/official_protocol_benchmark_manifest_20260604/missing_assets.csv
?? outputs/official_protocol_benchmark_manifest_20260604/next_actions.csv
?? outputs/official_protocol_benchmark_manifest_20260604/official_dataset_counts.csv
?? outputs/official_protocol_benchmark_manifest_20260604/official_split_inventory.csv
?? outputs/official_protocol_benchmark_manifest_20260604/report.md
?? outputs/official_protocol_benchmark_manifest_20260604/spaq_official_test_manifest.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/cache/huggingface/version.txt
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/dataset_metrics.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/filter_audit.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/next_actions.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_candidate_ranking.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_coverage_audit.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_metrics_datasetwise.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_metrics_overall.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_predictions.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_recommendation.json
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_safety_cases.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_vs_candidates_predictions.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/predictions.csv
?? outputs/official_protocol_mdiqa_signal_benchmark_20260604/report.md
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/coverage_audit.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/hybrid_ratio_grid_metrics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/missing_coverage_recovery_plan.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/model_signal_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/next_actions.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/prediction_cache_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/report.md
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/safety_cases.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/signal_metrics_datasetwise.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/signal_metrics_overall.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/signal_predictions_common.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run/signal_recommendation.json
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run_no_musiq/model_signal_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/base_run_no_musiq/prediction_cache_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/c6_cap_diagnostics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/c6_recommendation.json
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/c6_variant_metrics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/candidate_ranking_with_c6.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/next_actions.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/report.md
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/safety_cases_with_c6.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/signal_metrics_datasetwise_with_c6.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/signal_metrics_overall_with_c6.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/signal_predictions_common_with_c6.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/next_actions.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_c6_cap_diagnostics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_candidate_ranking.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_hybrid_ratio_grid_metrics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_safety_cases.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_coverage_audit.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_metrics_datasetwise.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_metrics_overall.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_predictions.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_recommendation.json
?? outputs/official_protocol_technical_signal_benchmark_20260604/official_test_compatible_manifest.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/report.md
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/coverage_audit.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/hybrid_ratio_grid_metrics.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/missing_coverage_recovery_plan.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/model_signal_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/next_actions.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/prediction_cache_inventory.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/report.md
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/safety_cases.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/signal_metrics_datasetwise.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/signal_metrics_overall.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/signal_predictions_common.csv
?? outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/signal_recommendation.json
?? outputs/paired_production_mdiqa_gpu_bottleneck_audit_20260603/report.md
?? outputs/paired_production_mdiqa_significance_20260603/bootstrap_distribution_summary.csv
?? outputs/paired_production_mdiqa_significance_20260603/manifest_audit.csv
?? outputs/paired_production_mdiqa_significance_20260603/paired_eval_manifest.csv
?? outputs/paired_production_mdiqa_significance_20260603/paired_predictions.csv
?? outputs/paired_production_mdiqa_significance_20260603/paired_significance_results.csv
?? outputs/paired_production_mdiqa_significance_20260603/production_prediction_summary.csv
?? outputs/paired_production_mdiqa_significance_20260603/production_scoring_source_audit.md
?? outputs/paired_production_mdiqa_significance_20260603/range_alignment_summary.csv
?? outputs/paired_production_mdiqa_significance_20260603/report.md
?? outputs/paired_production_mdiqa_significance_expanded_20260603/bootstrap_distribution_summary_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/next_actions.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/paired_eval_manifest_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/paired_significance_results_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/production_baseline_discrepancy_audit.md
?? outputs/paired_production_mdiqa_significance_expanded_20260603/production_prediction_summary_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/range_alignment_summary_expanded.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/report.md
?? outputs/paired_production_mdiqa_significance_expanded_20260603/runtime_estimate.md
?? outputs/paired_production_mdiqa_significance_expanded_20260603/sample_stability_summary.csv
?? outputs/paired_production_mdiqa_significance_expanded_20260603/tf_gpu_environment_audit.md
?? outputs/paper_protocol_technical_iqa_audit_20260530/next_action.md
?? outputs/paper_protocol_technical_iqa_audit_20260530/protocol_reproducibility_matrix.csv
?? outputs/paper_protocol_technical_iqa_audit_20260530/report.md
?? outputs/paq2piq_external_official_split_search_20260530/download_log.txt
?? outputs/paq2piq_external_official_split_search_20260530/external_split_candidates.csv
?? outputs/paq2piq_external_official_split_search_20260530/report.md
?? outputs/paq2piq_external_official_split_search_20260530/source_urls_checked.txt
?? outputs/paq2piq_flive_toolbox_split_verify_20260530/filename_match_report.csv
?? outputs/paq2piq_flive_toolbox_split_verify_20260530/meta_info_FLIVEDataset.csv
?? outputs/paq2piq_flive_toolbox_split_verify_20260530/split_count_summary.csv
?? outputs/paq2piq_flive_toolbox_split_verify_20260530/verification_report.md
?? outputs/paq2piq_official_split_search_20260530/candidate_split_files.csv
?? outputs/paq2piq_official_split_search_20260530/report.md
?? outputs/paq2piq_official_split_search_20260530/search_commands_used.txt
?? outputs/research_protocol_baseline_eval_audit_20260530/proposed_smoke_eval_plan.md
?? outputs/research_protocol_baseline_eval_audit_20260530/report.md
?? outputs/research_protocol_baseline_eval_audit_20260530/scoring_entrypoint_inventory.csv
?? outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/flive_image_smoke.csv
?? outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/koniq_run01_smoke.csv
?? outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/spaq_smoke.csv
?? outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/smoke_eval_commands.txt
?? outputs/research_protocol_bounded_eval_20260530/bounded_metrics.csv
?? outputs/research_protocol_bounded_eval_20260530/bounded_predictions.csv
?? outputs/research_protocol_bounded_eval_20260530/bounded_sample_manifest.csv
?? outputs/research_protocol_bounded_eval_20260530/commands_used.txt
?? outputs/research_protocol_bounded_eval_20260530/report.md
?? outputs/research_protocol_bounded_eval_20260530/runtime_summary.csv
?? outputs/research_protocol_bounded_eval_analysis_20260530/metric_interpretation.csv
?? outputs/research_protocol_bounded_eval_analysis_20260530/next_eval_recommendation.md
?? outputs/research_protocol_bounded_eval_analysis_20260530/report.md
?? outputs/research_protocol_image_test_eval_20260530/commands_used.txt
?? outputs/research_protocol_image_test_eval_20260530/image_test_manifest.csv
?? outputs/research_protocol_image_test_eval_20260530/image_test_metrics.csv
?? outputs/research_protocol_image_test_eval_20260530/image_test_predictions.csv
?? outputs/research_protocol_image_test_eval_20260530/report.md
?? outputs/research_protocol_image_test_eval_20260530/runtime_summary.csv
?? outputs/research_protocol_image_test_eval_20260530/score_range_checks.csv
?? outputs/research_protocol_image_test_eval_analysis_20260530/metric_interpretation.csv
?? outputs/research_protocol_image_test_eval_analysis_20260530/next_eval_recommendation.md
?? outputs/research_protocol_image_test_eval_analysis_20260530/report.md
?? outputs/research_protocol_image_test_eval_analysis_20260530/score_scale_diagnosis.csv
?? outputs/research_protocol_iqa_manifests_20260530/report.md
?? outputs/research_protocol_production_technical_eval_20260530/commands_used.txt
?? outputs/research_protocol_production_technical_eval_20260530/production_formula_audit.md
?? outputs/research_protocol_production_technical_eval_20260530/production_score_range_checks.csv
?? outputs/research_protocol_production_technical_eval_20260530/production_technical_metrics.csv
?? outputs/research_protocol_production_technical_eval_20260530/production_technical_predictions.csv
?? outputs/research_protocol_production_technical_eval_20260530/report.md
?? outputs/research_protocol_production_technical_eval_analysis_20260530/baseline_comparison.csv
?? outputs/research_protocol_production_technical_eval_analysis_20260530/calibration_diagnosis.csv
?? outputs/research_protocol_production_technical_eval_analysis_20260530/next_action.md
?? outputs/research_protocol_production_technical_eval_analysis_20260530/report.md
?? outputs/research_protocol_smoke_scoring_20260530/commands_used.txt
?? outputs/research_protocol_smoke_scoring_20260530/report.md
?? outputs/research_protocol_smoke_scoring_20260530/smoke_metrics.csv
?? outputs/research_protocol_smoke_scoring_20260530/smoke_predictions.csv
?? outputs/rgnet_paper_reproduction_preflight_20260526/report.md
?? outputs/student_flive_hybrid_benchmark_20260604/checklist.md
?? outputs/student_flive_hybrid_benchmark_20260604/context-notes.md
?? outputs/student_flive_hybrid_benchmark_20260604/coverage_audit.csv
?? outputs/student_flive_hybrid_benchmark_20260604/datasetwise_hybrid_metrics.csv
?? outputs/student_flive_hybrid_benchmark_20260604/hybrid_candidate_metrics.csv
?? outputs/student_flive_hybrid_benchmark_20260604/hybrid_candidate_predictions.csv
?? outputs/student_flive_hybrid_benchmark_20260604/hybrid_ratio_grid_metrics.csv
?? outputs/student_flive_hybrid_benchmark_20260604/hybrid_recommendation.json
?? outputs/student_flive_hybrid_benchmark_20260604/next_actions.csv
?? outputs/student_flive_hybrid_benchmark_20260604/report.md
?? outputs/student_flive_hybrid_benchmark_20260604/safety_cases.csv
?? outputs/student_training_manifest_50k_20260601/source_group_summary.csv
?? outputs/student_training_manifest_50k_20260601/student_split_dataset_decile_summary.csv
?? outputs/student_training_manifest_50k_20260601/student_split_dataset_summary.csv
?? outputs/student_training_manifest_50k_20260601/student_split_decile_summary.csv
?? outputs/student_training_manifest_50k_20260601/student_split_summary.csv
?? outputs/student_training_manifest_50k_20260601/student_training_manifest.csv
?? outputs/student_training_manifest_50k_20260601/student_training_manifest_report.md
?? outputs/student_training_manifest_design_20260601/report.md
?? outputs/student_training_manifest_design_20260601/student_split_dataset_decile_summary.csv
?? outputs/student_training_manifest_design_20260601/student_split_summary.csv
?? outputs/student_training_manifest_design_20260601/student_training_manifest.csv
?? outputs/student_training_manifest_design_20260601/student_training_manifest_report.md
?? outputs/technical_iqa_combo_benchmark_20260524/bootstrap_ci.csv
?? outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv
?? outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_hard_fp.csv
?? outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_koniq.csv
?? outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv
?? outputs/technical_iqa_combo_benchmark_20260524/hard_fp_summary.csv
?? outputs/technical_iqa_combo_benchmark_20260524/pairwise_combo_comparison.csv
?? outputs/technical_iqa_combo_benchmark_20260524/report.md
?? outputs/technical_iqa_combo_benchmark_20260524/summary_by_dataset.csv
?? outputs/technical_iqa_signal_comparison_20260604/checklist.md
?? outputs/technical_iqa_signal_comparison_20260604/context-notes.md
?? outputs/technical_iqa_signal_comparison_20260604/coverage_audit.csv
?? outputs/technical_iqa_signal_comparison_20260604/hybrid_ratio_grid_metrics.csv
?? outputs/technical_iqa_signal_comparison_20260604/missing_coverage_recovery_plan.csv
?? outputs/technical_iqa_signal_comparison_20260604/model_signal_inventory.csv
?? outputs/technical_iqa_signal_comparison_20260604/next_actions.csv
?? outputs/technical_iqa_signal_comparison_20260604/prediction_cache_inventory.csv
?? outputs/technical_iqa_signal_comparison_20260604/report.md
?? outputs/technical_iqa_signal_comparison_20260604/safety_cases.csv
?? outputs/technical_iqa_signal_comparison_20260604/signal_metrics_datasetwise.csv
?? outputs/technical_iqa_signal_comparison_20260604/signal_metrics_overall.csv
?? outputs/technical_iqa_signal_comparison_20260604/signal_predictions_common.csv
?? outputs/technical_iqa_signal_comparison_20260604/signal_recommendation.json
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/c6_cap_diagnostics.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/c6_recommendation.json
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/c6_variant_metrics.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/candidate_ranking_with_c6.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/checklist.md
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/context-notes.md
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/next_actions.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/report.md
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/safety_cases_with_c6.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/signal_metrics_datasetwise_with_c6.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/signal_metrics_overall_with_c6.csv
?? outputs/technical_iqa_signal_comparison_with_c6_20260604/signal_predictions_common_with_c6.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/bootstrap_ci.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/fusion_predictions_flive.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/fusion_predictions_hard_fp_v2.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/fusion_predictions_koniq.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/fusion_predictions_spaq.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/hard_fp_summary.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/pairwise_cap_comparison.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/report.md
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/summary_by_dataset.csv
?? outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/top_overscored_hard_fp.md
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/hard_fp_delta.csv
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/rank_delta_summary.csv
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/report.md
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/top_boosted_by_c6.csv
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/top_demoted_by_c6.csv
?? outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/topk_precision.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/dataset_aware_risk_matrix.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/hybrid_teacher_design.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/mdiqa_vs_production_significance_plan.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/musiq_official_vs_local_note.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/next_actions.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/production_technical_score_audit.csv
?? outputs/technical_teacher_hybrid_design_audit_20260603/report.md
?? outputs/thesis_material_audit_20260607/evidence_index.csv
?? outputs/thesis_material_audit_20260607/experiment_summary.csv
?? outputs/thesis_material_audit_20260607/missing_evidence.md
?? outputs/thesis_material_audit_20260607/report.md
?? outputs/thesis_material_audit_20260607/thesis_outline.md
?? outputs/thesis_material_correction_audit_20260607/alamp_comparison.md
?? outputs/thesis_material_correction_audit_20260607/corrected_report.md
?? outputs/thesis_material_correction_audit_20260607/corrected_thesis_claims.csv
?? outputs/thesis_material_correction_audit_20260607/individual_model_inventory.csv
?? outputs/thesis_material_correction_audit_20260607/stage5_reassessment.md
?? outputs/thesis_material_final_correction_20260607/final_corrected_report.md
?? outputs/thesis_material_final_correction_20260607/final_model_status.csv
?? outputs/thesis_material_final_correction_20260607/final_thesis_claims.csv
?? outputs/thesis_material_final_correction_20260607/final_thesis_outline.md
?? outputs/thesis_material_final_correction_20260607/icca_exclusion_note.md
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/gate_candidate_ranking.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_metrics_datasetwise.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_metrics_overall.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_modified_rows.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_predictions.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_recommendation.json
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/heuristic_gate_severe_fp_audit.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/image_quality_heuristic_metrics.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/next_actions.csv
?? outputs/topiq_flive_heuristic_gate_full_simulation_20260608/report.md
?? outputs/topiq_flive_safety_gate_simulation_20260608/gate_candidate_ranking.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/gate_metrics_datasetwise.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/gate_metrics_overall.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/gate_modified_rows.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/gate_recommendation.json
?? outputs/topiq_flive_safety_gate_simulation_20260608/gated_predictions.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/next_actions.csv
?? outputs/topiq_flive_safety_gate_simulation_20260608/report.md
?? outputs/topiq_flive_safety_gate_simulation_20260608/severe_fp_fix_audit.csv
?? outputs/topiq_flive_safety_signal_ablation_20260608/ablation_recommendation.json
?? outputs/topiq_flive_safety_signal_ablation_20260608/mdiqa_oracle_gate_metrics.csv
?? outputs/topiq_flive_safety_signal_ablation_20260608/next_actions.csv
?? outputs/topiq_flive_safety_signal_ablation_20260608/remaining_severe_fp_analysis.csv
?? outputs/topiq_flive_safety_signal_ablation_20260608/report.md
?? outputs/topiq_flive_safety_signal_ablation_20260608/simple_quality_metric_probe.csv
?? outputs/topiq_mixed112_artifact_recovery_20260527/report.md
?? outputs/topiq_mixed112_artifact_recovery_20260527/topiq_lite_mixed112.metadata.json
?? outputs/topiq_teacher_guard_design_20260531/false_negative_cases.csv
?? outputs/topiq_teacher_guard_design_20260531/false_positive_cases.csv
?? outputs/topiq_teacher_guard_design_20260531/guard_candidate_metrics.csv
?? outputs/topiq_teacher_guard_design_20260531/report.md
?? outputs/topiq_teacher_guard_holdout_20260601/errors.csv
?? outputs/topiq_teacher_guard_holdout_20260601/false_negative_cases.csv
?? outputs/topiq_teacher_guard_holdout_20260601/false_positive_cases.csv
?? outputs/topiq_teacher_guard_holdout_20260601/guarded_holdout_metrics.csv
?? outputs/topiq_teacher_guard_holdout_20260601/guarded_holdout_predictions.csv
?? outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
?? outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest_report.md
?? outputs/topiq_teacher_guard_holdout_20260601/predictions.csv
?? outputs/topiq_teacher_guard_holdout_20260601/report.md
?? outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv
?? outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest_report.md
?? outputs/topiq_teacher_label_dryrun_20260601/errors.csv
?? outputs/topiq_teacher_label_dryrun_20260601/false_negative_cases.csv
?? outputs/topiq_teacher_label_dryrun_20260601/false_positive_cases.csv
?? outputs/topiq_teacher_label_dryrun_20260601/guarded_label_metrics.csv
?? outputs/topiq_teacher_label_dryrun_20260601/guarded_labels.csv
?? outputs/topiq_teacher_label_dryrun_20260601/inference_report.md
?? outputs/topiq_teacher_label_dryrun_20260601/predictions.csv
?? outputs/topiq_teacher_label_dryrun_20260601/report.md
?? outputs/topiq_teacher_label_dryrun_20260601/score_histograms.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/errors.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/expansion_manifest.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/expansion_manifest_report.md
?? outputs/topiq_teacher_label_expansion_50k_20260601/false_negative_cases.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/false_positive_cases.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/guarded_label_metrics.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/guarded_labels.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/inference_report.md
?? outputs/topiq_teacher_label_expansion_50k_20260601/predictions.csv
?? outputs/topiq_teacher_label_expansion_50k_20260601/report.md
?? outputs/topiq_teacher_label_expansion_50k_20260601/score_histograms.csv
?? outputs/topiq_teacher_performance_audit_20260601/datasetwise_teacher_metrics.csv
?? outputs/topiq_teacher_performance_audit_20260601/missing_baseline_requirements.csv
?? outputs/topiq_teacher_performance_audit_20260601/report.md
?? outputs/topiq_teacher_performance_audit_20260601/teacher_metric_summary.csv
?? outputs/topiq_teacher_performance_audit_20260601/teacher_vs_student_failure_summary.csv
?? outputs/topiq_teacher_setup_audit_20260531.md
?? outputs/topiq_teacher_smoke_20260531/errors.csv
?? outputs/topiq_teacher_smoke_20260531/predictions.csv
?? outputs/topiq_teacher_smoke_20260531/report.md
?? outputs/topiq_teacher_smoke_20260531/smoke_manifest_30.csv
?? outputs/topiq_teacher_validation_20260531/errors.csv
?? outputs/topiq_teacher_validation_20260531/predictions.csv
?? outputs/topiq_teacher_validation_20260531/report.md
?? outputs/topiq_teacher_validation_20260531/validation_manifest.csv
?? outputs/topiq_teacher_validation_20260531/validation_manifest_report.md
?? plan.md
?? pytorch_gpu_check.py
?? reports/advisor_report/advisor_report.md
?? reports/advisor_report/audit_summary.md
?? reports/advisor_report/data_sources.md
?? reports/advisor_report/figure_plan.md
?? reports/advisor_report/figures/C_technical_quality_comparison.png
?? reports/advisor_report/figures/D_aesthetic_quality_comparison.png
?? reports/advisor_report/figures/E_score_distribution.png
?? reports/advisor_report/figures/F_prediction_scatter.png
?? reports/advisor_report/figures/G_hard_case_grid.png
?? reports/advisor_report/scripts/generate_figures.py
?? reports/advisor_report/tables/B_model_summary.csv
?? reports/advisor_report/tables/I_decision_matrix.csv
?? reports/advisor_weekly_report/README.md
?? reports/advisor_weekly_report/data_sources.md
?? reports/advisor_weekly_report/figure_plan.md
?? reports/advisor_weekly_report/figures/model_status.png
?? reports/advisor_weekly_report/figures/performance_srcc.png
?? reports/advisor_weekly_report/figures/timeline.png
?? reports/advisor_weekly_report/final_professor_report.md
?? reports/advisor_weekly_report/full_audit_index.md
?? reports/advisor_weekly_report/scripts/fix_figures.py
?? reports/advisor_weekly_report/scripts/generate_reports.py
?? reports/advisor_weekly_report/scripts/generate_tables_figures.py
?? reports/advisor_weekly_report/tables/experiment_timeline.csv
?? reports/advisor_weekly_report/tables/model_inventory.csv
?? reports/advisor_weekly_report/tables/success_failure_matrix.csv
?? reports/advisor_weekly_report/weekly/week_01.md
?? reports/advisor_weekly_report/weekly/week_02.md
?? reports/advisor_weekly_report/weekly/week_03.md
?? reports/advisor_weekly_report/weekly/week_04.md
?? reports/advisor_weekly_report/weekly/week_05.md
?? reports/advisor_weekly_report/weekly/week_06.md
?? reports/advisor_weekly_report/weekly/week_07.md
?? reports/advisor_weekly_report/weekly/week_08.md
?? reports/advisor_weekly_report/weekly/week_09.md
?? requirements-gemma4-distill.txt
?? run_evaluation.py
?? run_wsl_c6_parity_with_orientation.py
?? scripts/alamp_patch_audit_runner.py
?? scripts/analyze_acut_aesthetic_multiobjective_weights.py
?? scripts/analyze_alamp_4096_thresholds.py
?? scripts/audit_acut_aesthetic_benchmark_csvs.py
?? scripts/create_mixed_topiq_datasets.py
?? scripts/debug_gold_eval.py
?? scripts/eval_final_topiq_candidates_vs_existing_technical.py
?? scripts/eval_hard_fp_v2.py
?? scripts/eval_old_vs_new_alamp_rgnet_paper_benchmark.py
?? scripts/eval_rgnet_paper_aadb_tflite_parity.py
?? scripts/eval_techiqa_guard_v1_stage1.py
?? scripts/eval_topiq_rankcalib_v2_probe.py
?? scripts/eval_topiq_rankcalib_v2_stage3.py
?? scripts/eval_topiq_rankcalib_v2_stage3_probeB.py
?? scripts/eval_topiq_rankcalib_v2_stage3_probeC.py
?? scripts/eval_topiq_ranking_probe.py
?? scripts/eval_topiq_ranking_probe_lam01.py
?? scripts/eval_topiq_ranking_probe_lam02.py
?? scripts/evaluate_acut_final_ratio_core.py
?? scripts/evaluate_acut_product_gold_weights.py
?? scripts/evaluate_acut_product_gold_weights_vila.py
?? scripts/evaluate_fusion.py
?? scripts/evaluate_icaa_official_weight.py
?? scripts/export_icaa_torchscript_smoke.py
?? scripts/export_rgnet_paper_aadb_tflite.py
?? scripts/export_topiq_lite_ranking_lam01.py
?? scripts/parity_icaa_torchscript_real_images.py
?? scripts/run_threshold_calibration.py
?? scripts/smoke_test_icaa.py
?? scripts/smoke_test_icaa_real_images.py
?? scripts/smoke_test_icaa_weight_load.py
?? scripts/technical_iqa/analyze_topiq_teacher_guard_design.py
?? scripts/technical_iqa/apply_topiq_guarded_label_dryrun.py
?? scripts/technical_iqa/apply_topiq_guarded_label_expansion_50k.py
?? scripts/technical_iqa/audit_50k_mdiqa_manifest.py
?? scripts/technical_iqa/benchmark_student_flive_hybrid_20260604.py
?? scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py
?? scripts/technical_iqa/benchmark_technical_iqa_signals_with_c6_20260604.py
?? scripts/technical_iqa/build_hybrid_teacher_label_dryrun.py
?? scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py
?? scripts/technical_iqa/build_hybrid_teacher_label_dryrun_2k.py
?? scripts/technical_iqa/build_mdiqa_50k_smoke_manifest.py
?? scripts/technical_iqa/build_paired_significance_manifest.py
?? scripts/technical_iqa/build_unknown_fallback_dryrun.py
?? scripts/technical_iqa/consolidate_50k_manifest.py
?? scripts/technical_iqa/diagnose_musiq_failure.py
?? scripts/technical_iqa/diagnostic_performance.py
?? scripts/technical_iqa/diagnostic_performance_detailed.py
?? scripts/technical_iqa/evaluate_student_iqa_14105_checkpoints.py
?? scripts/technical_iqa/evaluate_topiq_guard_holdout.py
?? scripts/technical_iqa/export_student_iqa_14105_tflite.py
?? scripts/technical_iqa/identify_remaining_mdiqa_rows.py
?? scripts/technical_iqa/make_student_training_manifest.py
?? scripts/technical_iqa/make_student_training_manifest_50k.py
?? scripts/technical_iqa/make_topiq_guard_holdout_manifest.py
?? scripts/technical_iqa/make_topiq_label_dryrun_manifest.py
?? scripts/technical_iqa/make_topiq_label_expansion_50k_manifest.py
?? scripts/technical_iqa/make_topiq_teacher_validation_manifest.py
?? scripts/technical_iqa/production_component_audit.py
?? scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py
?? scripts/technical_iqa/run_hybrid_teacher_range_calibration_2k.py
?? scripts/technical_iqa/run_mdiqa_50k_smoke_inference.py
?? scripts/technical_iqa/run_mdiqa_correct_protocol_full_benchmark.py
?? scripts/technical_iqa/run_mdiqa_flive_image_test_sanity.py
?? scripts/technical_iqa/run_mdiqa_official_image_only_benchmark.py
?? scripts/technical_iqa/run_mdiqa_official_variant_sanity_benchmark.py
?? scripts/technical_iqa/run_mdiqa_remaining_full_inference.py
?? scripts/technical_iqa/run_mdiqa_sanity_benchmark.py
?? scripts/technical_iqa/run_mdiqa_student_gap_analysis.py
?? scripts/technical_iqa/run_mdiqa_variant_sanity_benchmark.py
?? scripts/technical_iqa/run_mdiqa_vs_production_paired_significance.py
?? scripts/technical_iqa/run_mdiqa_vs_production_paired_significance_expanded.py
?? scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py
?? scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark_fallback.py
?? scripts/technical_iqa/run_paired_production_predictions_for_mdiqa.py
?? scripts/technical_iqa/run_paired_production_predictions_for_mdiqa_expanded.py
?? scripts/technical_iqa/run_topiq_guard_holdout.py
?? scripts/technical_iqa/run_topiq_label_dryrun_inference.py
?? scripts/technical_iqa/run_topiq_label_expansion_50k_inference.py
?? scripts/technical_iqa/run_topiq_teacher_smoke.py
?? scripts/technical_iqa/run_topiq_teacher_validation.py
?? scripts/technical_iqa/score_fallback_audit_data.py
?? scripts/technical_iqa/simulate_heuristic_gates.py
?? scripts/technical_iqa/simulate_safety_gates.py
?? scripts/technical_iqa/simulate_safety_signal_ablation.py
?? scripts/technical_iqa/smoke_test_kaggle_musiq.py
?? scripts/technical_iqa/smoke_test_mdiqa.py
?? scripts/technical_iqa/test_musiq_fallbacks.py
?? scripts/technical_iqa/test_musiq_fallbacks_cpu.py
?? scripts/technical_iqa/test_tflite_parity.py
?? scripts/technical_iqa/train_mobile_student_iqa_14105_longer.py
?? scripts/technical_iqa/train_mobile_student_iqa_14105_short.py
?? scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py
?? scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py
?? scripts/technical_iqa/train_mobile_student_iqa_50k_label_scaling_smoke.py
?? scripts/technical_iqa/train_mobile_student_iqa_50k_ranking_smoke.py
?? scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py
?? scripts/technical_iqa/train_mobile_student_iqa_v1.py
?? scripts/technical_iqa/train_mobile_student_iqa_v2.py
?? scripts/technical_iqa/train_mobile_student_iqa_v3.py
?? scripts/technical_iqa/train_mobile_student_iqa_v4a.py
?? scripts/technical_iqa/train_mobile_student_iqa_v4b.py
?? scripts/technical_iqa/validate_unknown_fallback.py
?? severe_fp_dist.py
?? skills-lock.json
?? src/data/build_flive_csv.py
?? src/data/flive_utils.py
?? src/data/make_techiqa_guard_train_with_fp.py
?? src/data/prepare_flive.py
?? src/distill/__init__.py
?? src/distill/aggregation.py
?? src/distill/build_weighted_teacher_manifest.py
?? src/distill/check_teacher_consistency.py
?? src/distill/combine_teacher_labels.py
?? src/distill/compare_aesthetic_candidates.py
?? src/distill/compare_student_candidates.py
?? src/distill/config.py
?? src/distill/data.py
?? src/distill/evaluate_student.py
?? src/distill/evaluation.py
?? src/distill/export_student_tflite.py
?? src/distill/generate_teacher_labels.py
?? src/distill/modeling.py
?? src/distill/preprocessing.py
?? src/distill/train_student.py
?? src/distill/verify_student_tflite.py
?? src/eval/evaluate_alamp_paper_ava_v2_graph_lite.py
?? src/eval/evaluate_alamp_paper_mpnet_ava.py
?? src/gemma_distill/__init__.py
?? src/gemma_distill/audit_teacher.py
?? src/gemma_distill/build_dataset.py
?? src/gemma_distill/common.py
?? src/gemma_distill/evaluate_gemma.py
?? src/gemma_distill/generate_gemini_teacher.py
?? src/gemma_distill/merge_adapter.py
?? src/gemma_distill/predict_gemma.py
?? src/gemma_distill/quantize_export.py
?? src/gemma_distill/teacher.py
?? src/gemma_distill/train_gemma.py
?? src/infer/explanation_contract.py
?? src/infer/vila_full_local_worker.py
?? src/models/aesthetic_nima.py
?? src/models/alamp.py
?? src/models/alamp_dual_branch_teacher.py
?? src/models/alamp_paper_ava_v2_graph_lite.py
?? src/models/alamp_paper_fusion.py
?? src/models/alamp_paper_mpnet.py
?? src/models/mobile_alamp_v2.py
?? src/models/musiq.py
?? src/models/pairwise_comparator.py
?? src/models/rgnet.py
?? src/models/technical_regressor.py
?? src/models/topiq_rankcalib.py
?? src/preprocess/make_spaq_flive_csv.py
?? src/saliency/u2net_model.py
?? src/train/train_alamp_paper_ava_v2_graph_lite.py
?? src/train/train_alamp_paper_mpnet_ava.py
?? src/train/train_alamp_paper_mpnet_graph_fusion.py
?? src/train/train_mobile_alamp_v2.py
?? src/train/train_topiq_rankcalib_stage1.py
?? src/train/train_topiq_rankcalib_stage2_ranking.py
?? src/train/train_topiq_rankcalib_stage3_distill.py
?? stage_i_savedmodel_log.txt
?? stage_i_savedmodel_report.json
?? stage_j_tflite_fp32_log.txt
?? stage_j_tflite_fp32_report.json
?? test_erfc/variables/variables.data-00000-of-00001
?? test_erfc/variables/variables.index
?? test_flex.py
?? test_flex_v2.py
?? test_flex_v3.py
?? test_images/.gitkeep
?? test_images/README.md
?? test_outputs/distill_dry_run_probe/resolved_config.json
?? test_sampler_tflite.py
?? test_vila/Bad/1675564423029-24.jpg
?? test_vila/Bad/1675564423029-6.jpg
?? test_vila/Bad/1715855266134.jpg
?? test_vila/Bad/1720499657653.jpg
?? test_vila/Bad/20230201_181300.jpg
?? test_vila/Bad/20230201_181510.jpg
?? test_vila/Bad/20240101_143158.jpg
?? test_vila/Bad/20240502_193823.jpg
?? test_vila/Bad/20240505_153023.jpg
?? test_vila/Bad/20240506_120558.jpg
?? test_vila/Bad/20240510_213907.jpg
?? test_vila/Bad/20240517_060242.jpg
?? test_vila/Bad/20240517_060245.jpg
?? test_vila/Bad/20250129_054504.jpg
?? test_vila/Bad/20250129_102520.jpg
?? test_vila/Bad/20250129_103313.jpg
?? test_vila/Bad/IMG_2785.JPG
?? test_vila/Good/1675342165226-13.jpg
?? test_vila/Good/1675342165226-3.jpg
?? test_vila/Good/1675342165226-5.jpg
?? test_vila/Good/1704296944756.jpg
?? test_vila/Good/1720499666964.jpg
?? test_vila/Good/1720499666985.jpg
?? test_vila/Good/1738744324186(good_night).jpg
?? test_vila/Good/20231231_150055.jpg
?? test_vila/Good/20240509_154353.jpg
?? test_vila/Good/20240515_092725.jpg
?? test_vila/Good/20250204_133404.jpg
?? test_vila/Good/20250204_135520.jpg
?? test_vila/Good/20250204_163550.jpg
?? test_vila/Good/20250204_170733.jpg
?? test_vila/Good/20250204_174237(0).jpg
?? test_vila/Good/20250205_000515(good_night).jpg
?? test_vila/Good/20250205_072644.jpg
?? test_vila/Good/20250217_144845.jpg
?? test_vila/Good/IMG_20240519_215827_510.jpg
?? test_vila/Good/IMG_2781.JPG
?? test_vila/Soso/1675564423029-9.jpg
?? test_vila/Soso/1675564674752-15.jpg
?? test_vila/Soso/1675564674752-17.jpg
?? test_vila/Soso/1675564674752-2.jpg
?? test_vila/Soso/20240503_210355.jpg
?? test_vila/Soso/20240505_170541.jpg
?? test_vila/Soso/20240506_140415.jpg
?? test_vila/Soso/20240510_191716.jpg
?? test_vila/Soso/20240515_073118.jpg
?? test_vila/Soso/IMG_2729(soso-night_.JPG
?? test_vila/Soso/IMG_2738(soso-night_.JPG
?? test_vila/Soso/IMG_2739(soso-night_.JPG
?? test_vila/Soso/IMG_2748(soso-night_.JPG
?? tflite_input_sensitivity_report.json
?? tools/aesthetic_weight_lab/README.md
?? tools/aesthetic_weight_lab/checklist.md
?? tools/aesthetic_weight_lab/context-notes.md
?? tools/aesthetic_weight_lab/html_report.py
?? tools/aesthetic_weight_lab/model_registry.py
?? tools/aesthetic_weight_lab/plan.md
?? tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py
?? tools/aesthetic_weight_lab/tflite_model_runner.py
?? tools/audit_distortionguard_smoke.py
?? tools/audit_patch_sizes.py
?? tools/audit_patch_stats.py
?? tools/build_acut_human_review_package.py
?? tools/build_aesthetic_only_comparison.py
?? tools/build_raw_aesthetic_scores_html.py
?? tools/calibrate_mpnet_thresholds.py
?? tools/calibrate_rgnet_ava_threshold.py
?? tools/compare_alamp_patch_selectors.py
?? tools/compare_alamp_patch_selectors_v3.py
?? tools/compare_alamp_patch_selectors_v4.py
?? tools/compile_ablation_results.py
?? tools/diagnose_gemma_visual_grounding.py
?? tools/export_mobile_alamp_v2_tflite.py
?? tools/gemini_api_teacher_pipeline.py
?? tools/gemma4_bnb4_quantized_save.py
?? tools/gemma4_e4b_aesthetic3_expanded_pipeline.py
?? tools/gemma4_latency_probe.py
?? tools/generate_u2net_saliency_maps.py
?? tools/mine_hard_fp_candidates.py
?? tools/run_aesthetic_ensemble_experiment.py
?? tools/run_alamp_mpnet_compare.sh
?? tools/run_alamp_mpnet_compare_full.sh
?? tools/run_alamp_mpnet_smoke.sh
?? tools/run_alamp_paper_mpnet_patch_generation.sh
?? tools/run_alamp_paper_mpnet_patch_generation_v2.sh
?? tools/run_alamp_paper_strict_patch_selection.sh
?? tools/run_alamp_v3_selection.sh
?? tools/run_alamp_v4_selection.sh
?? tools/run_eval_only.py
?? tools/run_rgnet_paper_v1_ablation.py
?? tools/run_u2net_generation.sh
?? tools/run_v2_smoke.sh
?? tools/temp_download.py
?? tools/visualize_alamp_patch_boxes.py
?? v3_analysis.py
?? verify_teacher_safety.py
```

## Repository top level

```text
f	./run_evaluation.py
d	./test_erfc
f	./ds_decile.py
d	./docs
f	./nima_restart.log
f	./run_wsl_c6_parity_with_orientation.py
f	./rgnet_train.log
d	./reports
f	./.firebaserc
d	./forWeights
f	./list_models.py
d	./outputs
d	./photo_score_project
f	./severe_fp_dist.py
d	./src
f	./.codex
d	./models
d	./__pycache__
d	./forWeights2
f	./requirements.txt
f	./plan.md
f	./TROUBLESHOOTING - 복사본.md:Zone.Identifier
d	./checkpoints
d	./external
f	./GEMINI - 복사본.md:Zone.Identifier
f	./checkpoint_local_inventory.csv
f	./.gitignore
d	./.gemini
l	./ptxas
d	./tools
f	./train_sequence.log
f	./analyze_datasets.py
f	./mixed112_files.txt
f	./test_sampler.tflite
d	./.git
d	./test_vila
f	./gpu_check.py
d	./configs
f	./test_flex.py
d	./.venv_test
f	./pytorch_gpu_check.py
f	./test_flex_v3.py
f	./nima_full.log
f	./requirements-vila.txt
f	./missing_checkpoint_actions.csv
f	./pairwise_train.log
f	./verify_teacher_safety.py
d	./weights
d	./test_images
f	./codex_session.log
f	./generate_preprocessed_previews.py
f	./requirements-firebase-worker.txt
f	./context-notes.md
f	./skills-lock.json
f	./requirements-gemma4-distill.txt
f	./gemma4_second_pass.log
d	./.venv_mdiqa
d	./.venv
f	./finalize_csvs.py
d	./functions
f	./check_severe.py
f	./.env.example
f	./test_sampler_tflite.py
f	./checkpoint_drive_listing.csv
f	./checklist.md
f	./onnx2tf_full_help.txt
d	./MDIQA
f	./GEMINI.md
f	./stage_i_savedmodel_report.json
f	./firestore-debug.log
f	./storage.rules
f	./AGENTS.md
f	./stage_j_tflite_fp32_log.txt
d	./android_gemma_latency_harness
f	./diagnose_parity.py
d	./.venv_vila
f	./v3_analysis.py
f	./stage_j_tflite_fp32_report.json
d	./.venv_gpu
f	./.gitattributes
f	./firestore.rules
f	./musiq_train.log
f	./debug_savedmodel_nan.py
d	./scripts
f	./SETUP_LOCAL.md
d	./.venv_tf
f	./=0.45.0
f	./SETUP_WORKER.md
f	./onnx2tf_help.txt
f	./conversion_debug.log
f	./gemma4_dataset_expand.log
f	./checkpoint_mapping_candidates.csv
f	./firebase.json
d	./test_outputs
f	./test.tflite
d	./test_samples
f	./TROUBLESHOOTING.md
f	./requirements-gemini.txt
d	./experiments
d	./data
f	./alamp_train.log
f	./stage_i_savedmodel_log.txt
f	./README.md
f	./tflite_input_sensitivity_report.json
d	./.agents
d	./open_clip
f	./test_flex_v2.py
d	./exports
f	./inspect_sm_weights.py
```

## Directory tree

This is a directory-only tree through depth 3. The Git database, virtual-environment internals, `functions/node_modules`, and raw image-file descendants were pruned from this presentation because they are dependency/data payloads rather than repository structure. Their top-level roots remain visible, and model artifacts are inventoried separately.

```text
./test_erfc
./test_erfc/assets
./test_erfc/variables
./docs
./reports
./reports/advisor_weekly_report
./reports/advisor_weekly_report/tables
./reports/advisor_weekly_report/figures
./reports/advisor_weekly_report/weekly
./reports/advisor_weekly_report/scripts
./reports/advisor_report
./reports/advisor_report/tables
./reports/advisor_report/figures
./reports/advisor_report/scripts
./forWeights
./forWeights/outputs
./forWeights/outputs/paper_faithful_audit_20260509
./forWeights/models
./forWeights/models/aesthetic
./forWeights/tools
./forWeights/tools/aesthetic_weight_lab
./forWeights/.git
./forWeights/.git/info
./forWeights/.git/refs
./forWeights/.git/logs
./forWeights/.git/objects
./forWeights/.git/branches
./forWeights/.git/hooks
./forWeights/configs
./forWeights/test_images
./forWeights/.agents
./outputs
./outputs/mdiqa_flive_protocol_audit_20260601
./outputs/topiq_lite_layer_check
./outputs/rgnet_float32_retry_20260508
./outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain_aadb_finetune
./outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain
./outputs/rgnet_float32_retry_20260508/logs
./outputs/rgnet_float32_retry_20260508/eval
./outputs/rgnet_float32_retry_20260508/scripts
./outputs/rgnet_float32_retry_20260508/tflite
./outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_smoke
./outputs/alamp_paper_fusion_4096_graphlite_retry_v1
./outputs/alamp_paper_fusion_4096_graphlite_retry_v1/features
./outputs/aesthetic_ensemble_production_precheck_20260603
./outputs/firebase_worker_local_smoke
./outputs/firebase_worker_local_smoke/_worker_local
./outputs/mobile_student_iqa_v4a_20260601_smoke_fixed
./outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221456
./outputs/icaa_export_safe_onnx_20260514_210121
./outputs/test_vila_nvila_resume_20260415
./outputs/icaa_tf_native_savedmodel_20260516_125422
./outputs/probe_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_131131
./outputs/icaa_tf_native_savedmodel_20260516_044158
./outputs/distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523
./outputs/dryrun_distortionguard_stageB2_partial_unfreeze_b3_20260524
./outputs/alamp_object_graph_subset_20260511
./outputs/alamp_object_graph_subset_20260511/detections_conf015
./outputs/alamp_object_graph_subset_20260511/detections_conf010
./outputs/alamp_object_graph_subset_20260511/detections
./outputs/alamp_object_graph_subset_20260511/graphs
./outputs/alamp_object_graph_subset_20260511/graphs_conf010
./outputs/alamp_object_graph_subset_20260511/subsets
./outputs/icaa_tf_native_tflite_fp32_20260516_132108
./outputs/icaa_tf_native_tflite_fp32_20260516_140027
./outputs/nima_fixed_preproc_eval_20260525
./outputs/hybrid_teacher_adjusted_calibration_2k_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_125633
./outputs/saved_models
./outputs/distortionguard_stageA_synthetic_medium500_unfrozen_continue_e10_20260523
./outputs/icaa_export_safe_onnx_20260514_194704
./outputs/alamp_paper_faithfulness_audit_20260511
./outputs/acut_stage4_nima_musiq_alamp_rgnet
./outputs/acut_icaa_skip_weight_search_20260601
./outputs/acut_icaa_skip_weight_search_20260601/__pycache__
./outputs/smallrun_alamp_multipatch_teacher_20260524
./outputs/icaa_tf_native_tflite_fp32_20260516_130721
./outputs/icaa_export_safe_onnx_20260514_194438
./outputs/icaa_tf_native_savedmodel_20260516_132933
./outputs/icaa_tf_native_savedmodel_20260516_132933/saved_model
./outputs/icaa_tf_native_savedmodel_20260516_130941
./outputs/icaa_tf_native_savedmodel_20260516_130941/saved_model
./outputs/acut_product_vila_local_real
./outputs/eval_topiq_lite_ranking_probe_lam02_gap10_e10_20260517
./outputs/hybrid_teacher_unknown_fallback_audit_20260603
./outputs/probe1024_topiq_lite_flive_e15_frozen
./outputs/probe1024_topiq_lite_flive_e15_frozen/saved_model
./outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528
./outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/wsl_with_orientation
./outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/preprocess_debug
./outputs/topiq_teacher_label_expansion_50k_20260601
./outputs/distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524
./outputs/alamp_restart
./outputs/alamp_restart/saved_model
./outputs/distill_exec_20260411
./outputs/distill_exec_20260411/stage5_student_aadb_subset
./outputs/distill_exec_20260411/subset_manifests
./outputs/topiq_flive_heuristic_gate_full_simulation_20260608
./outputs/alamp_paper_fusion_1024_graphlite_20260513
./outputs/alamp_paper_fusion_1024_graphlite_20260513/features
./outputs/alamp_paper_fusion_1024_graphlite_20260513/graphlite
./outputs/alamp_paper_fusion_1024_graphlite_20260513/eval
./outputs/rgnet_old_vs_paper_fp16_compare_20260514
./outputs/rgnet_old_vs_paper_fp16_compare_20260514/scripts
./outputs/icaa_tf_native_savedmodel_20260516_125945
./outputs/icaa_tf_native_stage_f_20260516_113010
./outputs/icaa_tf_native_savedmodel_20260516_042006
./outputs/icaa_tf_native_savedmodel_20260516_042006/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_120948
./outputs/acut_product_gold_weight_eval_20260603
./outputs/technical_iqa_signal_comparison_20260604
./outputs/icaa_tf_native_tflite_fp32_20260516_132825
./outputs/ablate_vila_010
./outputs/acut_product_ready_smoke_post_integration
./outputs/pairwise_restart
./outputs/pairwise_restart/saved_model
./outputs/icaa_export_safe_v3_pytorch_parity_20260515_170845
./outputs/icaa_int8_qat_audit_20260601
./outputs/final_topiq_lite_mixed112_export_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_043548
./outputs/smoke_topiq_lite_spaq
./outputs/smoke_topiq_lite_spaq/saved_model
./outputs/icaa_tf_native_stage_g3_20260516_113510
./outputs/acut_weekly_research_journal_20260526
./outputs/acut_weekly_research_journal_20260526/figures
./outputs/full_iqa_iaa_model_design_audit_20260611
./outputs/icaa_tf_native_tflite_fp32_20260516_121121
./outputs/ablate_vila_015
./outputs/acut_product_vila_local_score_reasoning_v10_probe
./outputs/vila_scores_run
./outputs/rgnet_paper_aadb_tflite_export_20260514
./outputs/rgnet_paper_aadb_tflite_export_20260514/parity
./outputs/icaa_tf_native_stage_c_20260515_175817
./outputs/alamp_v4_full_ava_20260517
./outputs/alamp_v4_full_ava_20260517/saliency_maps
./outputs/alamp_v4_full_ava_20260517/visualizations
./outputs/alamp_v4_full_ava_20260517/subsets
./outputs/eval_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517
./outputs/icaa_export_safe_v2_onnx_20260514_215455
./outputs/firebase_worker_new_path_verify_gpu
./outputs/firebase_worker_new_path_verify_gpu/_worker_local
./outputs/smallrun_alamp_multipatch_teacher_classweighted_block5_20260524
./outputs/eval_topiq_lite_ranking_probe_lam05_e12_20260517
./outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525
./outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525/eval_best_valtest256
./outputs/gemini_api_teacher_aesthetic3_20260427_revised
./outputs/gemini_api_teacher_aesthetic3_20260427_revised/thumbs
./outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282
./outputs/gemini_api_teacher_aesthetic3_20260427_revised/visual_grounding_diagnostic
./outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282
./outputs/icaa_torchscript_parity_20260514_151359
./outputs/test_vila_aesthetic_direct_compare_20260504_153044
./outputs/test_vila_aesthetic_direct_compare_20260504_153044/thumbs
./outputs/distortionguard_stageB3_flive_weighted_diagnosis_20260524
./outputs/acut_product_nvila8b_probe_v5
./outputs/research_protocol_bounded_eval_20260530
./outputs/icaa_performance_reverification_20260525
./outputs/smoke_distortionguard_stageA_synthetic_medium500_e2_20260523
./outputs/full_topiq_lite_spaq_frozen_e30_20260517
./outputs/full_topiq_lite_spaq_frozen_e30_20260517/saved_model
./outputs/topiq_teacher_guard_design_20260531
./outputs/icaa_tf_native_savedmodel_20260516_134439
./outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model
./outputs/firebase_worker_normal_path_verify
./outputs/firebase_worker_normal_path_verify/_worker_local
./outputs/acut_product_vila_local_score_reasoning_v3
./outputs/icaa_tf_native_tflite_fp32_20260516_122927
./outputs/eval_tflite_technical_compare
./outputs/icaa_tf_native_savedmodel_20260516_124619
./outputs/icaa_tf_native_savedmodel_20260516_124619/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_131805
./outputs/icaa_tf_native_tflite_fp32_20260516_132937
./outputs/rgnet_v1_aadb_pil_resize_retrain_20260524_tflite
./outputs/dryrun_rgnet_v1_cascaded_denseaspp_aadb_20260525
./outputs/icaa_tf_native_stage_g1_20260516_113112
./outputs/alamp_paper_mpnet_design_audit_20260511
./outputs/alamp_paper_mpnet_design_audit_20260511/patch_visualizations
./outputs/alamp_paper_mpnet_design_audit_20260511/subsets
./outputs/ranking_topiq_lite_mixed112_flive_pairs_lam02_gap10_e10_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_132805
./outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609
./outputs/probe_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517
./outputs/icaa_tf_native_stage_b_20260515_174854
./outputs/icaa_tf_native_savedmodel_20260516_124135
./outputs/icaa_tf_native_savedmodel_20260516_124135/saved_model
./outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526
./outputs/icaa_tf_native_tflite_fp32_20260516_124410
./outputs/capstone_model_audit_20260711
./outputs/capstone_model_audit_20260711/00_inventory
./outputs/acut_product_ready_diversity_check
./outputs/dryrun_alamp_multipatch_teacher_sgd_default_compat_20260525
./outputs/acut_product_ready_smoke
./outputs/ranking_topiq_lite_mixed112_flive_pairs_lam05_e3_20260517
./outputs/acut_aesthetic_benchmark_csv_audit_20260603
./outputs/paired_production_mdiqa_gpu_bottleneck_audit_20260603
./outputs/icaa_official_eval_20260514_141732
./outputs/icaa_tf_native_savedmodel_20260516_123508
./outputs/icaa_tf_native_savedmodel_20260516_123508/saved_model
./outputs/icaa_tf_native_stage_g2_mlp_debug_20260516_031415
./outputs/acut_capstone_final_report_20260526
./outputs/acut_capstone_final_report_20260526/figures
./outputs/acut_capstone_final_report_20260526/parsed
./outputs/aesthetic_ensemble_weight_search_20260603
./outputs/aesthetic_ensemble_weight_search_20260603/__pycache__
./outputs/probe1024_topiq_lite_spaq_e15_unfrozen
./outputs/probe1024_topiq_lite_spaq_e15_unfrozen/saved_model
./outputs/rgnet_restart
./outputs/rgnet_restart/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_132453
./outputs/mobile_alamp_v2_pretrained_4096
./outputs/topiq_mixed112_artifact_recovery_20260527
./outputs/rgnet_v1_aadb_pil_resize_retrain_20260524
./outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/eval_best_pil_bilinear
./outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/eval_final_pil_bilinear
./outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/eval_final_pil_bilinear_check
./outputs/icaa_tf_native_stage_f_20260515_183424
./outputs/dryrun_techiqa_guard_v1_stage4_fpdata_check_20260520
./outputs/aesthetic_cross_predictions_for_weight_search_20260603
./outputs/aesthetic_cross_predictions_for_weight_search_20260603/__pycache__
./outputs/firebase_worker_local_smoke_post_integration
./outputs/firebase_worker_local_smoke_post_integration/_worker_local
./outputs/acut_product_vila_local_format_check_v4
./outputs/mobile_student_iqa_v4b_checkpoint_reeval_20260601
./outputs/icaa_tf_native_savedmodel_20260516_133929
./outputs/icaa_tf_native_savedmodel_20260516_133929/saved_model
./outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517/eval
./outputs/distortionguard_hard_fp_v2_failure_taxonomy_20260524
./outputs/distortionguard_hard_fp_v2_failure_taxonomy_20260524/contact_sheets
./outputs/topiq_flive_safety_gate_simulation_20260608
./outputs/alamp_paper_fusion_smoke_20260513_fix
./outputs/alamp_paper_fusion_smoke_20260513_fix/features
./outputs/acut_product_contract_check
./outputs/mobile_student_iqa_14105_smoke_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_121312
./outputs/icaa_tf_native_tflite_fp32_20260516_121625
./outputs/icaa_tf_native_savedmodel_20260516_121445
./outputs/icaa_tf_native_savedmodel_20260516_121445/saved_model
./outputs/mobile_alamp_v2_4096_smoke
./outputs/final_topiq_lite_ranking_lam01_export_20260517
./outputs/alamp_multipatch_teacher_classweighted_block5_full_20260524
./outputs/icaa_tf_native_savedmodel_20260516_123244
./outputs/icaa_tf_native_savedmodel_20260516_123244/saved_model
./outputs/student_flive_hybrid_benchmark_20260604
./outputs/mdiqa_variant_sanity_20260601
./outputs/mdiqa_variant_sanity_20260601/cache
./outputs/test_vila_raw_aesthetic_scores_html_20260424
./outputs/test_vila_raw_aesthetic_scores_html_20260424/thumbs
./outputs/mobile_student_iqa_v2_error_analysis_20260601
./outputs/icaa_tf_native_stage_h_full_dat_20260516_040004
./outputs/eval_distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524
./outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609
./outputs/icaa_tf_native_tflite_fp32_20260516_130846
./outputs/official_protocol_benchmark_manifest_20260604
./outputs/hybrid_teacher_label_dryrun_10k_20260603
./outputs/alamp_gpu_acceleration_audit_20260603
./outputs/firebase_worker_stage5_student_nvila_fix_verify
./outputs/firebase_worker_stage5_student_nvila_fix_verify/_worker_local
./outputs/smoke_topiq_rankcalib_v2_stage1_e3_20260517
./outputs/icaa_ptq_candidates_20260601
./outputs/icaa_ptq_candidates_20260601/review_20260601
./outputs/icaa_ptq_candidates_20260601/__pycache__
./outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427
./outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427/experiment_a_flash_20
./outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427/experiment_b_pro_20
./outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427/experiment_b_pro_.
./outputs/acut_product_contract_check_smoke_diversity
./outputs/icaa_export_safe_v3_onnx_20260515_171049
./outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525
./outputs/smallrun_alamp_dual_branch_teacher_4096_20260524
./outputs/smallrun_alamp_dual_branch_teacher_4096_20260524/eval_test
./outputs/icaa_export_safe_original_parity_20260514_153136
./outputs/gemma4_mobile_export_probe_20260429
./outputs/gemma4_mobile_export_probe_20260429/quantized_bnb4_merged_gemini_teacher_282
./outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282
./outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525
./outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525/eval_best_valtest256
./outputs/icaa_tf_native_tflite_fp32_20260516_130121
./outputs/stage5_runs
./outputs/stage5_runs/stage5_smoke_test
./outputs/stage5_runs/test_vila_stage5_reference_compare_20260504_135426
./outputs/icaa_tf_native_tflite_fp32_20260516_134314
./outputs/mobile_student_iqa_50k_smoke_20260608
./outputs/topiq_flive_safety_signal_ablation_20260608
./outputs/paper_protocol_technical_iqa_audit_20260530
./outputs/icaa_tf_native_tflite_fp32_20260516_131903
./outputs/techiqa_guard_v1_direct_stage5_rank002_fpguard005_20260520
./outputs/mdiqa_official_protocol_expansion_audit_20260603
./outputs/acut_product_nvila8b_probe
./outputs/icaa_tf_native_tflite_fp32_20260516_125317
./outputs/eval_alamp_multipatch_teacher_classweighted_block5_full_20260524
./outputs/musiq_kaggle_smoke_20260601
./outputs/alamp_paper_fusion_4096_chunked_graphlite
./outputs/alamp_paper_fusion_4096_chunked_graphlite/features
./outputs/alamp_paper_fusion_4096_chunked_graphlite/graphlite
./outputs/ablate_vila_refined_010
./outputs/icaa_tf_native_tflite_fp16_20260516_143721
./outputs/icaa_tf_native_tflite_fp32_20260516_130352
./outputs/icaa_tf_native_tflite_fp32_20260516_043326
./outputs/smallrun_alamp_multipatch_teacher_papercap4096_20260525
./outputs/icaa_export_torchscript_20260514_150436
./outputs/alamp_paper_fusion_4096_graphlite_bs2
./outputs/alamp_paper_fusion_4096_graphlite_bs2/features
./outputs/dryrun_rgnet_v1_hybrid_spatial_20260525
./outputs/acut_product_ready_smoke_contract_wrapper_skip
./outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609
./outputs/icaa_tf_native_savedmodel_20260516_123326
./outputs/icaa_tf_native_savedmodel_20260516_123326/saved_model
./outputs/ablate_vila_refined_005
./outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609
./outputs/ranking_pairgen_probe_gap010_tmp_20260517
./outputs/technical_iqa_combo_benchmark_20260524
./outputs/mobile_student_iqa_14105_tflite_export_20260604
./outputs/icaa_tf_native_tflite_fp32_20260516_132922
./outputs/alamp_rgnet_aadb_ava_comparison_20260513
./outputs/icaa_tf_native_tflite_fp32_20260516_132545
./outputs/icaa_tf_native_savedmodel_20260516_125914
./outputs/icaa_tf_native_savedmodel_20260516_125914/saved_model
./outputs/probe1024_topiq_lite_spaq_e15_frozen
./outputs/probe1024_topiq_lite_spaq_e15_frozen/saved_model
./outputs/alamp_paper_ava_v1_design_audit_20260511
./outputs/icaa_tf_native_stage_e_20260515_181850
./outputs/research_protocol_image_test_eval_20260530
./outputs/icaa_tf_native_stage_d_20260515_180922
./outputs/dryrun_alamp_dual_branch_teacher_20260524
./outputs/dryrun_alamp_dual_branch_teacher_20260524/smoke_train
./outputs/dryrun_alamp_multipatch_teacher_recipe_default_compat_20260525
./outputs/mobile_student_iqa_50k_controlled_b0_20260608
./outputs/eval_final_topiq_candidates_vs_existing_technical_20260520
./outputs/logs
./outputs/acut_model_core_training_code_excerpt_20260601
./outputs/acut_model_core_training_code_excerpt_20260601/snippets
./outputs/icaa_tf_native_savedmodel_20260516_130526
./outputs/student_training_manifest_design_20260601
./outputs/test_vila_aesthetic_model_audit_20260423
./outputs/eval_topiq_rankcalib_v2_stage1_regression_e15_20260517
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_smoke_fixed_20260517
./outputs/rgnet_model_lineage_audit_20260513
./outputs/eval_full_topiq_lite_spaq_replacement_20260517
./outputs/alamp_patch_fidelity_audit_20260525
./outputs/icaa_tf_native_stage_h_full_dat_20260516_040212
./outputs/hybrid_teacher_range_calibration_2k_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_132647
./outputs/color_aesthetics_audit_20260509
./outputs/icaa_tf_native_tflite_fp32_20260516_122056
./outputs/mdiqa_smoke_20260601
./outputs/mdiqa_smoke_20260601/cache
./outputs/icaa_tf_native_savedmodel_20260516_123925
./outputs/icaa_tf_native_savedmodel_20260516_123925/saved_model
./outputs/probe1024_topiq_lite_flive_e15_unfrozen
./outputs/probe1024_topiq_lite_flive_e15_unfrozen/saved_model
./outputs/acut_final_ratio_core_eval_20260603
./outputs/probe1024_musiq_flive
./outputs/probe1024_musiq_flive/saved_model
./outputs/alamp_dual_branch_gcn_failure_analysis_20260524
./outputs/mobile_student_iqa_v4b_20260601_sanity
./outputs/acut_stage5_full_with_pairwise
./outputs/alamp_paper_fusion_4096_graphlite_oom_partial_20260514_1339
./outputs/alamp_paper_fusion_4096_graphlite_oom_partial_20260514_1339/features
./outputs/dryrun_alamp_multipatch_teacher_20260524
./outputs/acut_product_vila_local_probe
./outputs/official_protocol_mdiqa_signal_benchmark_20260604
./outputs/official_protocol_mdiqa_signal_benchmark_20260604/cache
./outputs/dryrun_rgnet_v1_cascaded_denseaspp_paper_recipe_20260526
./outputs/stage5_with_vila_reasoning
./outputs/icaa_tf_native_stage_g2_20260516_024326
./outputs/icaa_tf_native_tflite_fp32_20260516_134508
./outputs/firebase_worker_new_path_verify
./outputs/firebase_worker_new_path_verify/_worker_local
./outputs/known_dataset_full_label_dryrun_20260603
./outputs/icaa_tf_native_savedmodel_20260516_121053
./outputs/icaa_tf_native_savedmodel_20260516_121053/saved_model
./outputs/eval_topiq_lite_mixed112_unfreeze_replacement_20260517
./outputs/c6_android_wsl_parity_exif_fixed_20260528
./outputs/dryrun_rgnet_v1_aadb_pil_resize_retrain_20260524
./outputs/dryrun_rgnet_v1_aadb_pil_resize_retrain_20260524/eval_smoke
./outputs/eval_topiq_lite_flive_calibration_replacement_20260517
./outputs/acut_product_vila_local_score_reasoning_v9
./outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608
./outputs/dryrun_alamp_multipatch_teacher_classweighted_block5_20260524
./outputs/icaa_tf_native_tflite_fp32_20260516_114655
./outputs/research_protocol_image_test_eval_analysis_20260530
./outputs/eval_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
./outputs/icaa_tf_native_stage_h_full_dat_20260516_040045
./outputs/icaa_tf_native_tflite_fp32_20260516_130645
./outputs/acut_product_vila_local_score_reasoning_v6
./outputs/alamp_paper_fusion_smoke_20260513_finalguard
./outputs/alamp_paper_fusion_smoke_20260513_finalguard/features
./outputs/midrun_alamp_multipatch_teacher_papercap4096_8192_20260525
./outputs/mobile_alamp_v2_pretrained_smoke
./outputs/aesthetic_mobile_paper_benchmark_20260603
./outputs/aesthetic_mobile_paper_benchmark_20260603/scripts
./outputs/aesthetic_mobile_paper_benchmark_20260603/full
./outputs/aesthetic_mobile_paper_benchmark_20260603/smoke
./outputs/mobile_alamp_v2_pretrained_probe_1024
./outputs/alamp_detector_selection_20260511
./outputs/acut_product_nvila8b_probe_v10_gpu
./outputs/icaa_tf_native_tflite_fp32_20260516_132741
./outputs/rgnet_paper_aadb_mobile_export_audit_20260513
./outputs/musiq_restart
./outputs/musiq_restart/saved_model
./outputs/nima_underperformance_diagnosis_20260525
./outputs/icaa_tf_native_savedmodel_20260516_125453
./outputs/icaa_tf_native_savedmodel_20260516_125453/saved_model
./outputs/research_protocol_smoke_scoring_20260530
./outputs/icaa_torchscript_parity_20260514_151034
./outputs/techiqa_guard_v1_direct_stage2_partial_unfreeze_b3_flive15_20260520
./outputs/icaa_tf_native_savedmodel_20260516_125309
./outputs/icaa_tf_native_savedmodel_20260516_125309/saved_model
./outputs/ava_pretrain_aadb_finetune_20260507
./outputs/ava_pretrain_aadb_finetune_20260507/alamp_ava_pretrain_aadb_finetune
./outputs/ava_pretrain_aadb_finetune_20260507/logs
./outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune
./outputs/ava_pretrain_aadb_finetune_20260507/eval
./outputs/ava_pretrain_aadb_finetune_20260507/scripts
./outputs/ava_pretrain_aadb_finetune_20260507/tflite
./outputs/alamp_external_patch_validation_20260524
./outputs/probe1024_topiq_lite_koniq_e15_unfrozen
./outputs/probe1024_topiq_lite_koniq_e15_unfrozen/saved_model
./outputs/icaa_tf_native_savedmodel_20260516_130441
./outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524
./outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526
./outputs/icaa_export_safe_torchscript_20260514_195503
./outputs/alamp_paper_mpnet_patch_selector_v2_20260511
./outputs/alamp_paper_mpnet_patch_selector_v2_20260511/patch_visualizations
./outputs/alamp_paper_mpnet_patch_selector_v2_20260511/subsets
./outputs/alamp_paper_ava_v2_graph_lite_20260511
./outputs/alamp_paper_ava_v2_graph_lite_20260511/eval_mid
./outputs/alamp_paper_ava_v2_graph_lite_20260511/mid_train
./outputs/alamp_paper_ava_v2_graph_lite_20260511/smoke_train
./outputs/acut_product_vila_local_score_reasoning_v4
./outputs/ava_retrain_completion_audit_20260507
./outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549
./outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549/__pycache__
./outputs/icaa_tf_native_stage_f_20260515_182930
./outputs/dryrun_rgnet_v1_default_context_compat_20260525
./outputs/mixed_replay_topiq_lite_112_unfrozen_lr1e6_e12_20260517
./outputs/mixed_replay_topiq_lite_112_unfrozen_lr1e6_e12_20260517/saved_model
./outputs/eval_techiqa_guard_v1_stage4b_fpconfirmed_v2_lam005_20260522
./outputs/icaa_tf_native_savedmodel_20260516_130030
./outputs/icaa_tf_native_savedmodel_20260516_130030/saved_model
./outputs/mdiqa_cuda_diagnosis_20260601
./outputs/gemini_api_teacher_aesthetic3_20260427
./outputs/gemini_api_teacher_aesthetic3_20260427/thumbs
./outputs/icaa_torchscript_parity_20260514_151144
./outputs/acut_product_vila_local_score_reasoning_v2
./outputs/mdiqa_checkpoint_source_audit_20260601
./outputs/mdiqa_student_final_no_go_report_20260611
./outputs/mobile_student_iqa_v3_error_analysis_20260601
./outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525
./outputs/nima_ava_restart
./outputs/nima_ava_restart/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_132629
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526/final_model
./outputs/techiqa_guard_v1_direct_stage5_rank001_fpguard05_20260520
./outputs/icaa_tf_native_tflite_fp32_20260516_042524
./outputs/acut_transparency_quick_export_check
./outputs/stage5_selector_only
./outputs/icaa_tf_native_stage_h_full_dat_20260516_113539
./outputs/icaa_tf_native_stage_f_20260516_113458
./outputs/stage5_with_vila_reasoning_refined
./outputs/icaa_export_safe_v2_pytorch_parity_20260514_215433
./outputs/mobile_student_iqa_v4b_20260601_smoke_fixed
./outputs/musiq_kaggle_sanity_fallback_20260601
./outputs/ablate_vila_refined_000
./outputs/c6_topiq_tensor_source_dimension_match_20260529
./outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke
./outputs/alamp_paper_fusion_smoke_20260513
./outputs/alamp_paper_fusion_smoke_20260513/features
./outputs/acut_product_contract_check_diversity
./outputs/ablate_vila_000
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/eval
./outputs/icaa_tf_native_stage_g3_20260516_113200
./outputs/alamp_layout_aware_teacher_plan_20260524
./outputs/icaa_tf_native_stage_f_20260515_183750
./outputs/icaa_tf_native_savedmodel_20260516_125059
./outputs/icaa_tf_native_savedmodel_20260516_125059/saved_model
./outputs/mpnet_gpu_env_audit_20260512
./outputs/research_protocol_bounded_eval_analysis_20260530
./outputs/full_topiq_lite_unified_to_flive_calib_frozen_e20_20260517
./outputs/full_topiq_lite_unified_to_flive_calib_frozen_e20_20260517/saved_model
./outputs/dryrun_techiqa_guard_v1_stage2_partial_unfreeze_20260520
./outputs/alamp_paper_strict_patch_selector_20260512
./outputs/alamp_paper_strict_patch_selector_20260512/overlay_visualizations
./outputs/topiq_lite_replacement_export_feasibility_20260517
./outputs/eval_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517
./outputs/midrun_alamp_multipatch_teacher_papercap4096_sgd1e4_8192_20260525
./outputs/midrun_alamp_multipatch_teacher_papercap4096_sgd1e3_8192_20260525
./outputs/icaa_tflite_conversion_20260514_200500
./outputs/icaa_tflite_conversion_20260514_200500/tf_converter_debug
./outputs/icaa_tflite_conversion_20260514_200500/onnx_tf
./outputs/icaa_tflite_conversion_20260514_200500/no_flex
./outputs/icaa_tflite_conversion_20260514_200500/opset16
./outputs/icaa_tflite_conversion_20260514_200500/tf_converter_k_image
./outputs/icaa_tflite_conversion_20260514_200500/fdosm
./outputs/icaa_tflite_conversion_20260514_200500/no_flex_k_image
./outputs/icaa_tflite_conversion_20260514_200500/no_flex_b1
./outputs/icaa_tflite_conversion_20260514_200500/tf_converter_ois
./outputs/icaa_tflite_conversion_20260514_200500/tf2onnx
./outputs/icaa_tflite_conversion_20260514_200500/tf_converter
./outputs/probe_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
./outputs/eval_topiq_unified_full_flive_replacement_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_132044
./outputs/alamp_paper_mpnet_patch_selector_v4_20260512
./outputs/alamp_paper_mpnet_patch_selector_v4_20260512/patch_visualizations
./outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets
./outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke
./outputs/hybrid_teacher_label_design_review_2k_20260603
./outputs/acut_product_vila_local_score_reasoning_v11_probe
./outputs/alamp_dual_branch_gcn_teacher_preflight_20260524
./outputs/topiq_teacher_performance_audit_20260601
./outputs/eval_techiqa_guard_v1_stage5_rank001_fpguard05_20260520
./outputs/icaa_tf_native_tflite_fp32_20260516_134634
./outputs/paper_level_stage1_benchmark_alignment_20260523
./outputs/mdiqa_correct_protocol_full_benchmark_20260601
./outputs/mdiqa_correct_protocol_full_benchmark_20260601/cache
./outputs/icaa_tf_native_savedmodel_20260516_123646
./outputs/icaa_tf_native_savedmodel_20260516_123646/saved_model
./outputs/eval_alamp_multipatch_teacher_full_ava_20260524_smoke
./outputs/probe1024_topiq_lite_koniq_e15_frozen
./outputs/probe1024_topiq_lite_koniq_e15_frozen/saved_model
./outputs/technical_iqa_signal_comparison_with_c6_20260604
./outputs/icaa_tf_native_tflite_fp32_20260516_132223
./outputs/alamp_mpnet_vs_aadb_alamp_review_20260513
./outputs/final_topiq_mixed112_replacement_decision_20260520
./outputs/dryrun_rgnet_v1_default_recipe_compat_20260526
./outputs/dryrun_rgnet_v1_default_recipe_compat_20260526/eval
./outputs/c6_android_wsl_parity_exif_fixed_20260529
./outputs/c6_android_wsl_parity_exif_fixed_20260529/__pycache__
./outputs/distortionguard_stageA_synthetic_medium500_unfrozen_e5_20260523
./outputs/alamp_external_patch_full_conversion_20260524
./outputs/icaa_tf_native_tflite_fp32_20260516_115302
./outputs/rgnet_paper_aadb_regression_20260509
./outputs/rgnet_paper_aadb_regression_20260509/full_train
./outputs/rgnet_paper_aadb_regression_20260509/eval
./outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2
./outputs/rgnet_paper_aadb_regression_20260509/smoke_train
./outputs/rgnet_paper_aadb_regression_20260509/smoke_eval_v2
./outputs/test_vila_aesthetic_direct_compare_20260504_135316
./outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run
./outputs/test_vila_aesthetic_direct_compare_20260504_135316/thumbs
./outputs/mobile_student_iqa_v4b_20260601_smoke
./outputs/hybrid_teacher_label_design_review_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_131503
./outputs/icaa_tf_native_tflite_fp32_20260516_121509
./outputs/mdiqa_student_gap_analysis_20260608
./outputs/capstone_technical_iqa_baseline_summary_20260530
./outputs/icaa_tf_native_savedmodel_20260516_043231
./outputs/icaa_tf_native_savedmodel_20260516_043231/saved_model
./outputs/icaa_tf_native_savedmodel_20260516_115347
./outputs/icaa_tf_native_savedmodel_20260516_115347/saved_model
./outputs/rgnet_paper_faithfulness_next_step_20260525
./outputs/student_conservative_aesthetic_full_aadb_20260423
./outputs/student_conservative_aesthetic_full_aadb_20260423/logs
./outputs/student_conservative_aesthetic_full_aadb_20260423/comparison
./outputs/student_conservative_aesthetic_full_aadb_20260423/export
./outputs/acut_product_backend_cleanup_smoke
./outputs/icaa_tf_native_savedmodel_20260516_122031
./outputs/icaa_tf_native_savedmodel_20260516_122031/saved_model
./outputs/eval_topiq_lite_vs_mobile_baselines_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_123037
./outputs/topiq_teacher_smoke_20260531
./outputs/nima_ava_full
./outputs/icaa_tf_native_tflite_fp32_20260516_130947
./outputs/acut_stage5_full_with_pairwise_fast
./outputs/icaa_tf_native_tflite_fp32_20260516_132243
./outputs/techiqa_guard_v1_direct_stage4b_fpconfirmed_v2_lam005_20260522
./outputs/icaa_tf_native_savedmodel_20260516_130840
./outputs/icaa_tf_native_savedmodel_20260516_130840/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_123819
./outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523
./outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523/grids
./outputs/mdiqa_student_pre_final_audit_20260609
./outputs/icaa_qat_attempt_20260601
./outputs/icaa_qat_attempt_20260601/__pycache__
./outputs/thesis_material_audit_20260607
./outputs/audit_model_paper_basis_20260506
./outputs/alamp_paper_mpnet_selector_compare_20260512
./outputs/alamp_paper_mpnet_selector_compare_20260512/tiny_two_phase_smoke_20260513
./outputs/alamp_paper_mpnet_selector_compare_20260512/tiny_smoke
./outputs/alamp_paper_mpnet_selector_compare_20260512/train
./outputs/alamp_paper_mpnet_selector_compare_20260512/train_test
./outputs/alamp_paper_mpnet_selector_compare_20260512/eval
./outputs/alamp_paper_mpnet_selector_compare_20260512/smoke
./outputs/known_dataset_full_label_coverage_audit_20260603
./outputs/acut_product_vila_local_format_check
./outputs/icaa_tf_native_stage_g0_20260516_014248
./outputs/icaa_export_safe_torchscript_20260514_153437
./outputs/icaa_tf_native_savedmodel_20260516_130155
./outputs/icaa_tf_native_savedmodel_20260516_130155/saved_model
./outputs/eval_topiq_lite_spaq_to_koniq_replacement_20260517
./outputs/techiqa_guard_v1_direct_stage4_fpdata_lam005_20260520
./outputs/icaa_tf_native_tflite_fp32_20260516_132720
./outputs/mobile_student_iqa_14105_checkpoint_eval_20260604
./outputs/icaa_tf_native_savedmodel_20260516_050514
./outputs/smoke_topiq_rankcalib_v2_stage3_distill_lam1_rank002_e3_20260517
./outputs/icaa_tf_native_savedmodel_20260516_123605
./outputs/icaa_tf_native_savedmodel_20260516_123605/saved_model
./outputs/icaa_export_safe_onnx_20260514_210640
./outputs/test_vila_aesthetic_ensemble_experiment_20260423
./outputs/alamp_external_repo_audit_20260511
./outputs/distill_conservative_aesthetic_full_aadb_20260423
./outputs/distill_conservative_aesthetic_full_aadb_20260423/logs
./outputs/acut_product_nvila8b_probe_v7_gpu
./outputs/student_training_manifest_50k_20260601
./outputs/smoke_techiqa_guard_v1_direct_e3_20260520
./outputs/rgnet_paper_v1_ablation_20260509
./outputs/rgnet_paper_v1_ablation_20260509/eval_mid
./outputs/rgnet_paper_v1_ablation_20260509/mid
./outputs/rgnet_paper_v1_ablation_20260509/smoke
./outputs/tech_iqa_guard_v1_dataset_design_20260520
./outputs/icaa_export_safe_full_eval_20260514_193609
./outputs/mobile_student_iqa_v3_20260601_smoke
./outputs/alamp_multipatch_teacher_improvement_plan_20260524
./outputs/test_vila_aesthetic_direct_compare_20260504_153732
./outputs/test_vila_aesthetic_direct_compare_20260504_153732/thumbs
./outputs/probe1024_musiq_koniq
./outputs/probe1024_musiq_koniq/saved_model
./outputs/c6_android_parity_final_audit_20260530
./outputs/mobile_student_iqa_14105_smoke_gpu_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_132021
./outputs/rgnet_v1_cascaded_denseaspp_aadb_full_20260525
./outputs/mobile_student_iqa_v1_20260601_smoke
./outputs/acut_product_ready_smoke_contract
./outputs/acut_product_gold_weight_eval_test_vila_20260603
./outputs/acut_product_vila_local_format_check_v3
./outputs/gemini_api_teacher_aesthetic3_highspec_100_batch_20260427
./outputs/gemini_api_teacher_aesthetic3_highspec_100_batch_20260427/thumbs
./outputs/stage5_comparisons
./outputs/stage5_comparisons/stage5_compare_smoke
./outputs/alamp_paper_mpnet_patch_selector_v3_20260512
./outputs/alamp_paper_mpnet_patch_selector_v3_20260512/saliency_maps
./outputs/alamp_paper_mpnet_patch_selector_v3_20260512/saliency_maps_4096
./outputs/alamp_paper_mpnet_patch_selector_v3_20260512/patch_visualizations
./outputs/alamp_paper_mpnet_patch_selector_v3_20260512/subsets
./outputs/research_protocol_production_technical_eval_analysis_20260530
./outputs/topiq_student_distill
./outputs/mobile_student_iqa_v2_20260601_smoke
./outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
./outputs/acut_product_vila_local_format_check_v7
./outputs/icaa_official_eval_20260514_141838
./outputs/paper_faithful_audit_20260509
./outputs/icaa_tf_native_tflite_fp32_20260516_125534
./outputs/icaa_tf_native_stage_g1_20260516_022555
./outputs/mobile_alamp_v2_full_ava_20260519
./outputs/alamp_multipatch_teacher_full_ava_20260524
./outputs/smoke_musiq_koniq
./outputs/smoke_musiq_koniq/saved_model
./outputs/alamp_patch_overlay_audit_20260513
./outputs/alamp_patch_overlay_audit_20260513/val_overlays
./outputs/alamp_patch_overlay_audit_20260513/train_overlays
./outputs/final_c6_log_only_readiness_20260528
./outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608
./outputs/icaa_export_safe_original_parity_20260514_153239
./outputs/acut_product_vila_local_format_check_v5
./outputs/icaa_tf_native_savedmodel_20260516_121244
./outputs/icaa_tf_native_savedmodel_20260516_121244/saved_model
./outputs/alamp_paper_fusion_4096_chunked_graphlite_smoke
./outputs/alamp_paper_fusion_4096_chunked_graphlite_smoke/features
./outputs/alamp_paper_fusion_4096_chunked_graphlite_smoke/graphlite
./outputs/icaa_tf_native_stage_f_20260516_113327
./outputs/techiqa_guard_v1_direct_stage3_fpguard_lam005_20260520
./outputs/topiq_teacher_guard_holdout_20260601
./outputs/eval_rgnet_v1_cascaded_denseaspp_aadb_full_20260525
./outputs/techiqa_guard_v1_hard_fp_mining_20260522
./outputs/techiqa_guard_v1_hard_fp_mining_20260522/user_confirmed
./outputs/icaa_export_safe_original_parity_20260514_185821
./outputs/acut_product_nvila8b_probe_v2
./outputs/test_vila_aesthetic_direct_compare_20260504_153222
./outputs/test_vila_aesthetic_direct_compare_20260504_153222/thumbs
./outputs/ablate_vila_refined_015
./outputs/icaa_tf_native_tflite_fp32_20260516_131223
./outputs/icaa_tf_native_savedmodel_20260516_130549
./outputs/test_vila_worker_verify_full_20260416
./outputs/test_vila_worker_verify_full_20260416/_worker_local
./outputs/acut_product_gemini_backend_sample
./outputs/icaa_export_safe_onnx_20260514_195608
./outputs/aesthetic_mobile_paper_benchmark_20260603_audit
./outputs/icaa_tf_native_tflite_fp32_20260516_130042
./outputs/stage5_with_vila_reasoning_smoke_refined
./outputs/smoke_distortionguard_stageB_authentic_from_stageA_e3_20260523
./outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520
./outputs/icaa_tf_native_tflite_fp32_20260516_042626
./outputs/dryrun_distortionguard_stageA_init_weights_20260523
./outputs/icaa_tf_native_savedmodel_20260516_123215
./outputs/icaa_tf_native_savedmodel_20260516_123215/saved_model
./outputs/icaa_tf_native_savedmodel_20260516_122852
./outputs/icaa_tf_native_savedmodel_20260516_122852/saved_model
./outputs/mdiqa_distillation_50k_manifest_audit_20260608
./outputs/mixed_replay_topiq_lite_111_frozen_e15_20260517
./outputs/mixed_replay_topiq_lite_111_frozen_e15_20260517/saved_model
./outputs/dryrun_rgnet_ava_repro_full_1ep_20260526
./outputs/ava_retrain_rgnet_alamp_20260506
./outputs/ava_retrain_rgnet_alamp_20260506/rgnet_ava_unit_smoke
./outputs/ava_retrain_rgnet_alamp_20260506/post_train_eval
./outputs/ava_retrain_rgnet_alamp_20260506/logs
./outputs/ava_retrain_rgnet_alamp_20260506/eval_scripts
./outputs/ava_retrain_rgnet_alamp_20260506/alamp_ava_unit_smoke
./outputs/ava_retrain_rgnet_alamp_20260506/baseline_eval
./outputs/ava_retrain_rgnet_alamp_20260506/rgnet_ava_unit
./outputs/ava_retrain_rgnet_alamp_20260506/alamp_ava_unit
./outputs/ava_retrain_rgnet_alamp_20260506/subsets
./outputs/ava_retrain_rgnet_alamp_20260506/tflite
./outputs/thesis_material_final_correction_20260607
./outputs/paq2piq_official_split_search_20260530
./outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221739
./outputs/icaa_tf_native_savedmodel_20260516_125026
./outputs/distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524
./outputs/alamp_papercap_recipe_audit_20260525
./outputs/ablate_vila_005
./outputs/alamp_external_integration_feasibility_audit_20260524
./outputs/mpnet_gpu_cudnn_audit_20260513
./outputs/alamp_dual_branch_4096_comparison_20260524
./outputs/alamp_dual_branch_4096_comparison_20260524/eval_dual_branch_gcn_4096
./outputs/alamp_dual_branch_4096_comparison_20260524/eval_multipatch_only_4096
./outputs/official_protocol_technical_signal_benchmark_20260604
./outputs/official_protocol_technical_signal_benchmark_20260604/base_run
./outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test
./outputs/official_protocol_technical_signal_benchmark_20260604/base_run_no_musiq
./outputs/official_protocol_technical_signal_benchmark_20260604/c6_run
./outputs/icaa_tf_native_savedmodel_20260516_125701
./outputs/icaa_tf_native_savedmodel_20260516_125701/saved_model
./outputs/mdiqa_distillation_50k_full_targets_20260608
./outputs/mdiqa_distillation_50k_full_targets_20260608/cache
./outputs/mobile_alamp_v2_full_ava_20260519_tflite
./outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024
./outputs/icaa_tf_native_savedmodel_20260516_130422
./outputs/acut_product_ready_smoke_diversity_post_integration
./outputs/rgnet_paper_v1_ablation_full_candidates_20260510
./outputs/rgnet_paper_v1_ablation_full_candidates_20260510/smoke_combined
./outputs/rgnet_paper_v1_ablation_full_candidates_20260510/full_train
./outputs/rgnet_paper_v1_ablation_full_candidates_20260510/eval
./outputs/mobile_student_iqa_v2_20260601_fixed
./outputs/icaa_tf_native_stage_g0_20260516_113059
./outputs/model_strategy_review_20260506
./outputs/eval_techiqa_guard_v1_stage3_fpguard_lam005_20260520
./outputs/acut_product_nvila8b_probe_v8_gpu
./outputs/icaa_tf_native_tflite_fp32_20260516_134032
./outputs/eval_techiqa_guard_v1_stage5_rank002_fpguard005_v2_20260520
./outputs/icaa_tf_native_savedmodel_20260516_113654
./outputs/icaa_tf_native_savedmodel_20260516_113654/saved_model
./outputs/alamp_paper_fusion_1024_comparison_20260513
./outputs/final_c6_topiq_koniq_technical_score_decision_20260527
./outputs/acut_stage3_nima_musiq_alamp
./outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604
./outputs/mobile_student_iqa_v1_20260601
./outputs/mobile_student_iqa_v4a_20260601_smoke
./outputs/final_techiqa_guard_v1_nogo_20260522
./outputs/icaa_tf_native_savedmodel_20260516_121604
./outputs/icaa_tf_native_savedmodel_20260516_121604/saved_model
./outputs/probe1024_musiq_spaq
./outputs/probe1024_musiq_spaq/saved_model
./outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609
./outputs/icaa_export_safe_onnx_20260514_194833
./outputs/paper_level_aesthetic_gap_audit_20260523
./outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist03_rank0_e3_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_132300
./outputs/icaa_tf_native_stage_g3_dattention_debug_20260516_034800
./outputs/icaa_tf_native_stage_f_20260515_183945
./outputs/ranking_topiq_lite_mixed112_flive_pairs_lam05_e12_20260517
./outputs/paired_production_mdiqa_significance_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_121933
./outputs/acut_product_firebase_regression
./outputs/acut_product_vila_local_score_reasoning_v10
./outputs/aesthetic_weight_lab_structure_audit_20260508
./outputs/mixed_replay_topiq_lite_221_frozen_e15_20260517
./outputs/mixed_replay_topiq_lite_221_frozen_e15_20260517/saved_model
./outputs/dryrun_distortionguard_stageB3_flive_weighted_20260524
./outputs/eval_techiqa_guard_v1_stage5_rank002_fpguard005_20260520
./outputs/koniq_android_wsl_parity_variant_diagnosis_20260528
./outputs/mdiqa_sanity_20260601
./outputs/mdiqa_sanity_20260601/cache
./outputs/rgnet_mobile_resize_finetune_preflight_20260524
./outputs/icaa_tf_native_savedmodel_20260516_130104
./outputs/icaa_tf_native_savedmodel_20260516_130104/saved_model
./outputs/mdiqa_wsl_feasibility_audit_20260601
./outputs/icaa_export_safe_onnx_20260514_211156
./outputs/acut_product_nvila8b_probe_v3
./outputs/icaa_tf_native_tflite_fp32_20260516_124204
./outputs/nima_icaa_paper_level_eval_20260525
./outputs/icaa_tf_native_stage_g3_20260516_033136
./outputs/mobile_alamp_v2_4096_gpu_smoke
./outputs/icaa_tf_native_tflite_fp32_20260516_132136
./outputs/icaa_tf_native_tflite_fp32_20260516_125153
./outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526
./outputs/rgnet_faithful_architecture_audit_20260524
./outputs/icaa_tf_native_savedmodel_20260516_134232
./outputs/icaa_tf_native_savedmodel_20260516_134232/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_132524
./outputs/icaa_export_safe_onnx_20260514_210025
./outputs/alamp_paper_fusion_1024_graphgcn_20260513
./outputs/alamp_paper_fusion_1024_graphgcn_20260513/graphgcn
./outputs/alamp_paper_fusion_1024_graphgcn_20260513/features
./outputs/alamp_paper_fusion_1024_graphgcn_20260513/eval
./outputs/icaa_tf_native_state_dict_inspection
./outputs/icaa_export_safe_torchscript_20260514_195550
./outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517
./outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517/saved_model
./outputs/final_research_protocol_technical_iqa_baseline_report_20260530
./outputs/distill_gpu_runtime_check
./outputs/test_vila_aesthetic_only_stage5_current_conservative_20260423
./outputs/mobile_student_iqa_v2_training_loop_verify_20260601
./outputs/research_protocol_iqa_manifests_20260530
./outputs/acut_product_vila_local_score_reasoning_v8
./outputs/acut_product_ready_smoke_contract_wrapper
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/final_model
./outputs/firebase_worker_local_smoke_diversity
./outputs/firebase_worker_local_smoke_diversity/_worker_local
./outputs/eval_techiqa_guard_v1_stage4_fpdata_lam005_20260520
./outputs/icaa_tf_native_tflite_fp32_20260516_125926
./outputs/icaa_tf_native_savedmodel_20260516_122748
./outputs/icaa_tf_native_savedmodel_20260516_122748/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_130316
./outputs/icaa_export_safe_original_parity_20260514_153215
./outputs/icaa_tf_native_savedmodel_20260516_044408
./outputs/acut_weekly_research_journal_v2_20260526
./outputs/acut_weekly_research_journal_v2_20260526/topics
./outputs/acut_weekly_research_journal_v2_20260526/figures
./outputs/acut_weekly_research_journal_v2_20260526/weeks
./outputs/acut_weekly_research_journal_v2_20260526/parsed
./outputs/musiq_kaggle_sanity_20260601
./outputs/gemma4_e4b_aesthetic3_expanded_20260427
./outputs/gemma4_e4b_aesthetic3_expanded_20260427/train_run
./outputs/gemma4_e4b_aesthetic3_expanded_20260427/export
./outputs/gemma4_e4b_aesthetic3_expanded_20260427/evaluation
./outputs/gemma4_e4b_aesthetic3_expanded_20260427/teacher_chunks
./outputs/acut_model_core_code_audit_20260601
./outputs/technical_teacher_hybrid_design_audit_20260603
./outputs/dryrun_rgnet_v1_hybrid_spatial_default_compat_20260525
./outputs/alamp_paper_mpnet_4096_expansion
./outputs/alamp_paper_mpnet_4096_expansion/v4_4096
./outputs/acut_product_vila_local_format_check_v6
./outputs/rgnet_paper_ava_classification_20260510
./outputs/rgnet_paper_ava_classification_20260510/smoke_eval
./outputs/rgnet_paper_ava_classification_20260510/threshold_calibration
./outputs/rgnet_paper_ava_classification_20260510/mid_train
./outputs/rgnet_paper_ava_classification_20260510/mid_eval
./outputs/rgnet_paper_ava_classification_20260510/full_train
./outputs/rgnet_paper_ava_classification_20260510/eval
./outputs/rgnet_paper_ava_classification_20260510/smoke_train
./outputs/eval_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517
./outputs/icaa_export_safe_v3_tflite_probe_20260515_171113
./outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/onnx2tf_no_flex
./outputs/acut_product_vila_local_score_reasoning_v7
./outputs/icaa_tf_native_savedmodel_20260516_130340
./outputs/icaa_tf_native_savedmodel_20260516_130340/saved_model
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526
./outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/final_model
./outputs/mobile_alamp_v2_1024_probe
./outputs/acut_product_nvila8b_probe_v7
./outputs/topiq_lite_batch_check
./outputs/smoke_musiq_flive
./outputs/smoke_musiq_flive/saved_model
./outputs/paq2piq_external_official_split_search_20260530
./outputs/smoke_topiq_rankcalib_v2_stage2_flive_rank_lam005_e3_20260517
./outputs/midrun_alamp_multipatch_teacher_papercap4096_recipe_8192_20260525
./outputs/acut_product_nvila8b_probe_v9_gpu
./outputs/mobile_student_iqa_v4b_20260601
./outputs/icaa_tf_native_savedmodel_20260516_125005
./outputs/icaa_tf_native_savedmodel_20260516_121906
./outputs/icaa_tf_native_savedmodel_20260516_121906/saved_model
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517/full_train
./outputs/acut_product_vila_local_score_reasoning_v5
./outputs/acut_product_vila_local_real2
./outputs/musiq_kaggle_failure_diagnosis_20260601
./outputs/rgnet_paper_v1_aadb_regression_20260509
./outputs/rgnet_paper_v1_aadb_regression_20260509/smoke_eval
./outputs/rgnet_paper_v1_aadb_regression_20260509/mid_train
./outputs/rgnet_paper_v1_aadb_regression_20260509/full_train
./outputs/rgnet_paper_v1_aadb_regression_20260509/eval
./outputs/rgnet_paper_v1_aadb_regression_20260509/smoke_train_v2
./outputs/rgnet_paper_v1_aadb_regression_20260509/smoke_train
./outputs/mobile_student_iqa_v4a_20260601
./outputs/eval_topiq_lite_mixed_replay_replacement_20260517
./outputs/eval_distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524
./outputs/mdiqa_official_variant_sanity_20260601
./outputs/mdiqa_official_variant_sanity_20260601/cache
./outputs/musiq_wsl_kaggle_audit_20260601
./outputs/icaa_export_safe_onnx_20260514_195654
./outputs/c6_android_wsl_parity_20260527
./outputs/smoke_techiqa_guard_v1_build
./outputs/mobile_student_iqa_14105_short_train_20260604
./outputs/techiqa_guard_v1_direct_stage1_frozen_20260520
./outputs/eval_alamp_multipatch_teacher_full_ava_20260524
./outputs/distortionguard_iqa_v1_design_20260523
./outputs/test_vila_human_acut_review_stage5_vs_ensemble_20260423
./outputs/icaa_export_safe_original_parity_20260514_195312
./outputs/topiq_teacher_validation_20260531
./outputs/alamp_salient_detector_audit_20260512
./outputs/icaa_export_safe_original_parity_20260514_153053
./outputs/icaa_tf_native_savedmodel_20260516_130639
./outputs/icaa_tf_native_savedmodel_20260516_130639/saved_model
./outputs/paper_level_teacher_reproduction_plan_20260524
./outputs/thesis_material_correction_audit_20260607
./outputs/aesthetic_ensemble_weight_search_complete_20260603_audit
./outputs/icaa_export_safe_onnx_20260514_202941
./outputs/acut_product_nvila8b_probe_v6
./outputs/mobile_student_iqa_14105_full_controlled_20260604
./outputs/icaa_tf_native_savedmodel_20260516_125620
./outputs/icaa_tf_native_savedmodel_20260516_125620/saved_model
./outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist05_rank0005_e3_20260517
./outputs/icaa_export_safe_torchscript_20260514_153413
./outputs/acut_product_vila_local_score_reasoning_v1
./outputs/mdiqa_distillation_50k_smoke_targets_20260608
./outputs/mdiqa_distillation_50k_smoke_targets_20260608/cache
./outputs/icaa_tf_native_tflite_fp32_20260516_115414
./outputs/test_vila_aesthetic_direct_compare_20260504_153546
./outputs/test_vila_aesthetic_direct_compare_20260504_153546/thumbs
./outputs/flutter_aesthetic_metadata_20260523
./outputs/icaa_tf_native_savedmodel_20260516_130252
./outputs/icaa_tf_native_savedmodel_20260516_130252/saved_model
./outputs/test_vila_nvila_eval_20260415_214918
./outputs/test_vila_nvila_eval_20260415_214918/_worker_local
./outputs/aesthetic_full_paper_benchmark_gap_analysis_20260603_audit
./outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517
./outputs/rgnet_tflite_retry_20260507
./outputs/rgnet_tflite_retry_20260507/logs
./outputs/rgnet_tflite_retry_20260507/eval
./outputs/rgnet_tflite_retry_20260507/scripts
./outputs/rgnet_tflite_retry_20260507/tflite
./outputs/eval_techiqa_guard_v1_stage1_frozen_20260520
./outputs/mobile_alamp_v2_pretrained_4096_tflite
./outputs/aesthetic_weight_lab_tflite_tool_20260508
./outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke
./outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke
./outputs/alamp_paper_fusion_4096_chunked_graphgcn
./outputs/alamp_paper_fusion_4096_chunked_graphgcn/graphgcn
./outputs/alamp_paper_fusion_4096_chunked_graphgcn/features
./outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522
./outputs/alamp_paper_ava_classification_20260511
./outputs/alamp_paper_ava_classification_20260511/smoke_eval
./outputs/alamp_paper_ava_classification_20260511/threshold_calibration
./outputs/alamp_paper_ava_classification_20260511/mid_train
./outputs/alamp_paper_ava_classification_20260511/mid_eval
./outputs/alamp_paper_ava_classification_20260511/smoke_train
./outputs/icaa_export_safe_original_parity_20260514_153033
./outputs/acut_transparency_test_vila_20260416_gpu
./outputs/smallrun_alamp_multipatch_teacher_repeatfix_20260524
./outputs/hybrid_teacher_final_label_scope_decision_20260603
./outputs/icaa_tf_native_tflite_fp32_20260516_125713
./outputs/icaa_tf_native_savedmodel_20260516_123424
./outputs/icaa_tf_native_savedmodel_20260516_123424/saved_model
./outputs/icaa_export_torchscript_20260514_144117
./outputs/nima_preprocessing_fix_20260525
./outputs/nima_quantized_vs_existing_comparison_20260525
./outputs/mobile_student_iqa_14105_longer_controlled_20260604
./outputs/test_vila_worker_verify_top10_20260416
./outputs/test_vila_worker_verify_top10_20260416/_worker_local
./outputs/dryrun_alamp_multipatch_teacher_default_compat_20260525
./outputs/c6_topiq_preprocess_backend_ablation_20260529
./outputs/acut_stage1_nima
./outputs/icaa_tf_native_tflite_fp32_20260516_132355
./outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608
./outputs/alamp_paper_fusion_smoke_threshold_fix
./outputs/alamp_paper_fusion_smoke_threshold_fix/features
./outputs/rgnet_paper_reproduction_preflight_20260526
./outputs/research_protocol_production_technical_eval_20260530
./outputs/icaa_tf_native_savedmodel_20260516_130506
./outputs/icaa_export_safe_full_eval_20260514_193638
./outputs/mobile_student_iqa_v3_20260601
./outputs/distill_full_aadb
./outputs/icaa_export_safe_full_eval_20260514_190201
./outputs/eval_topiq_rankcalib_v2_stage3_hparam_probe_20260517
./outputs/icaa_official_eval_20260514_142417
./outputs/c6_topiq_tensor_fingerprint_compare_20260529
./outputs/aesthetic_paper_benchmark_old_vs_new_20260523
./outputs/acut_transparency_test_vila_20260416
./outputs/icaa_tf_native_tflite_fp32_20260516_131636
./outputs/tflite
./outputs/icaa_tf_native_savedmodel_20260516_123735
./outputs/icaa_tf_native_savedmodel_20260516_123735/saved_model
./outputs/acut_full_stack
./outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517
./outputs/techiqa_guard_v1_direct_stage5_rank002_fpguard005_v2_20260520
./outputs/icaa_tf_native_tflite_fp16_20260516_144023
./outputs/icaa_export_torchscript_20260514_144141
./outputs/smoke_musiq_spaq
./outputs/smoke_musiq_spaq/saved_model
./outputs/icaa_tf_native_savedmodel_20260516_115119
./outputs/icaa_tf_native_savedmodel_20260516_115119/saved_model
./outputs/icaa_tf_native_tflite_fp32_20260516_124642
./outputs/paq2piq_flive_toolbox_split_verify_20260530
./outputs/paq2piq_flive_toolbox_split_verify_20260530/.cache
./outputs/icaa_tf_native_tflite_fp32_20260516_124012
./outputs/mdiqa_flive_image_test_sanity_20260601
./outputs/mdiqa_flive_image_test_sanity_20260601/cache
./outputs/icaa_tf_native_savedmodel_20260516_043428
./outputs/icaa_tf_native_savedmodel_20260516_043428/saved_model
./outputs/acut_product_vila_local_format_check_v2
./outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525
./outputs/acut_model_training_code_audit_20260601
./outputs/eval_distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523
./outputs/icaa_tf_native_tflite_fp32_20260516_135745
./outputs/acut_aesthetic_weight_multiobjective_20260603
./outputs/acut_stage2_nima_musiq
./outputs/mobile_alamp_v2_full_ava_smoke_20260519
./outputs/icaa_tf_native_savedmodel_20260516_125735
./outputs/acut_verify_now
./outputs/hybrid_teacher_adjusted_calibration_design_20260603
./outputs/audit_aesthetic_model_origin_20260504
./outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524
./outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/parsed
./outputs/probe_topiq_rankcalib_v2_stage1_regression_e15_20260517
./outputs/aesthetic_teacher_reproduction_decision_20260526
./outputs/topiq_teacher_smoke
./outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist03_rank0005_e3_20260517
./outputs/icaa_tf_native_tflite_fp32_20260516_130213
./outputs/probe_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517
./outputs/paired_production_mdiqa_significance_expanded_20260603
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_saved_model_pseudo_gelu
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_keep_nhwc
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_tf_converter_keep_nchw
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_saved_model
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_tf_converter
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_keep_nchw
./outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_no_onnxsim
./outputs/icaa_tf_native_tflite_fp32_20260516_132950
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_smoke
./outputs/rgnet_paper_v1_aadb_finetune_ava_prior_smoke
./outputs/icaa_export_safe_original_parity_20260514_194000
./outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603
./outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive
./outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/smoke_adaptive
./outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/diagnostic_1000
./outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/scripts
./outputs/aesthetic_ensemble_weight_search_complete_20260603
./outputs/aesthetic_ensemble_weight_search_complete_20260603/__pycache__
./outputs/student_distillation_manifest_14105_20260603
./outputs/icaa_export_safe_torchscript_20260514_185840
./outputs/mobile_student_iqa_v2_20260601
./outputs/mobile_student_iqa_v2_training_loop_audit_20260601
./outputs/icaa_tf_native_savedmodel_20260516_124346
./outputs/icaa_tf_native_savedmodel_20260516_124346/saved_model
./outputs/gemma4_e4b_aesthetic3_smoke
./outputs/gemma4_e4b_aesthetic3_smoke/audit_prepass2
./outputs/gemma4_e4b_aesthetic3_smoke/audit_pass2
./outputs/gemma4_e4b_aesthetic3_smoke/evaluation_pass2b
./outputs/gemma4_e4b_aesthetic3_smoke/export_pass2b
./outputs/gemma4_e4b_aesthetic3_smoke/train_run
./outputs/gemma4_e4b_aesthetic3_smoke/export
./outputs/gemma4_e4b_aesthetic3_smoke/evaluation
./outputs/gemma4_e4b_aesthetic3_smoke/evaluation_pass2
./outputs/gemma4_e4b_aesthetic3_smoke/train_run_pass2b
./outputs/gemma4_e4b_aesthetic3_smoke/train_run_pass2
./outputs/icaa_tf_native_stage_f_20260516_112834
./outputs/acut_product_nvila8b_probe_v4
./outputs/icaa_tf_native_tflite_fp32_20260516_132602
./outputs/predictions
./outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221722
./outputs/research_protocol_baseline_eval_audit_20260530
./outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep
./outputs/full_topiq_lite_spaq_to_koniq_frozen_e30_20260517
./outputs/full_topiq_lite_spaq_to_koniq_frozen_e30_20260517/saved_model
./outputs/rgnet_paper_consolidated_20260511
./outputs/topiq_teacher_label_dryrun_20260601
./photo_score_project
./photo_score_project/test_outputs
./photo_score_project/test_outputs/validation_with_api_key
./src
./src/models
./src/models/__pycache__
./src/__pycache__
./src/configs
./src/infer
./src/infer/__pycache__
./src/datasets
./src/datasets/__pycache__
./src/train
./src/train/__pycache__
./src/preprocess
./src/preprocess/__pycache__
./src/gemma_distill
./src/gemma_distill/__pycache__
./src/export
./src/export/__pycache__
./src/firebase
./src/firebase/__pycache__
./src/vila
./src/vila/__pycache__
./src/distill
./src/distill/__pycache__
./src/eval
./src/eval/__pycache__
./src/data_prep
./src/data_prep/__pycache__
./src/utils
./src/utils/__pycache__
./src/saliency
./src/saliency/__pycache__
./src/data
./src/data/__pycache__
./models
./models/detectors
./models/aesthetic
./models/litert
./models/litert/gemma-4-E4B-it-litert-lm
./models/saliency
./__pycache__
./forWeights2
./forWeights2/outputs
./forWeights2/outputs/aesthetic_weight_lab_4models_run2
./forWeights2/outputs/aesthetic_weight_lab_4models
./forWeights2/outputs/paper_faithful_audit_20260509
./forWeights2/models
./forWeights2/models/aesthetic
./forWeights2/tools
./forWeights2/tools/aesthetic_weight_lab
./forWeights2/configs
./forWeights2/test_images
./forWeights2/.agents
./checkpoints
./checkpoints/technical
./checkpoints/nima_ava_gpu_fixed_preproc_20260525
./checkpoints/nima_ava_gpu_fixed_preproc_20260525/saved_model
./checkpoints/composition_aadb_gpu
./checkpoints/composition_aadb_gpu/saved_model
./checkpoints/fusion
./checkpoints/nima_ava_gpu
./checkpoints/nima_ava_gpu/saved_model
./checkpoints/NVILA-8B
./checkpoints/NVILA-8B/.cache
./checkpoints/NVILA-8B/llm
./checkpoints/NVILA-8B/vision_tower
./checkpoints/NVILA-8B/mm_projector
./checkpoints/technical_flive_patch_gpu
./checkpoints/technical_flive_patch_gpu/saved_model
./checkpoints/technical_flive_gpu
./checkpoints/student_full_aadb
./checkpoints/student_full_aadb/saved_model
./checkpoints/student_full_aadb/evaluation
./checkpoints/aesthetic
./checkpoints/rgnet_aadb_gpu
./checkpoints/rgnet_aadb_gpu/saved_model
./checkpoints/student_conservative_aesthetic_full_aadb_20260423
./checkpoints/student_conservative_aesthetic_full_aadb_20260423/saved_model
./checkpoints/student_conservative_aesthetic_full_aadb_20260423/evaluation
./checkpoints/technical_koniq_gpu
./checkpoints/technical_koniq_gpu/saved_model
./checkpoints/nima_ava
./checkpoints/ranking
./checkpoints/stage5_student_full_run
./checkpoints/stage5_student_full_run/saved_model
./checkpoints/stage5_student_full_run/evaluation
./checkpoints/alamp_aadb_gpu
./checkpoints/alamp_aadb_gpu/saved_model
./checkpoints/musiq_aadb_gpu
./checkpoints/musiq_aadb_gpu/saved_model
./checkpoints/pairwise_aadb_gpu
./checkpoints/pairwise_aadb_gpu/saved_model
./checkpoints/composition
./checkpoints/composition_aadb
./checkpoints/VILA1.5-3b
./checkpoints/VILA1.5-3b/.cache
./checkpoints/VILA1.5-3b/llm
./checkpoints/VILA1.5-3b/vision_tower
./checkpoints/VILA1.5-3b/mm_projector
./checkpoints/technical_flive_image_gpu
./checkpoints/technical_flive_image_gpu/saved_model
./external
./external/A-Lamp_external_audit
./external/A-Lamp_external_audit/A-Lamp
./external/icaa_official_repo
./external/icaa_official_repo/ICAA17K_code
./external/icaa_official_repo/SPAQ_code
./external/icaa_official_repo/.git
./external/topiq_teacher
./.gemini
./.gemini/skills
./.gemini/skills/to-issues
./.gemini/skills/grill-me
./.gemini/skills/improve-codebase-architecture
./.gemini/skills/handoff
./.gemini/skills/to-prd
./.gemini/skills/grill-with-docs
./tools
./tools/__pycache__
./tools/alamp_graph
./tools/alamp_graph/__pycache__
./tools/aesthetic_weight_lab
./tools/aesthetic_weight_lab/__pycache__
./test_vila
./test_vila/Soso
./test_vila/Bad
./test_vila/Good
./configs
./configs/paper_benchmarks
./configs/paper_benchmarks/rgnet_v1_ablations
./configs/paper_benchmarks/rgnet_v1_full_candidates
./weights
./weights/icaa_official
./test_images
./functions
./functions/src
./functions/scripts
./functions/lib
./MDIQA
./MDIQA/docs
./MDIQA/options
./MDIQA/options/test
./MDIQA/options/train
./MDIQA/checkpoints
./MDIQA/pyiqa
./MDIQA/pyiqa/models
./MDIQA/pyiqa/__pycache__
./MDIQA/pyiqa/archs
./MDIQA/pyiqa/losses
./MDIQA/pyiqa/matlab_utils
./MDIQA/pyiqa/utils
./MDIQA/pyiqa/data
./MDIQA/pyiqa/metrics
./MDIQA/.git
./MDIQA/.git/info
./MDIQA/.git/refs
./MDIQA/.git/logs
./MDIQA/.git/objects
./MDIQA/.git/branches
./MDIQA/.git/hooks
./MDIQA/ResultsCalibra
./MDIQA/ResultsCalibra/ref_dir
./MDIQA/ResultsCalibra/dist_dir
./MDIQA/datasets
./MDIQA/datasets/PARA
./MDIQA/datasets/koniq10k
./MDIQA/datasets/SPAQ
./MDIQA/datasets/meta_info
./MDIQA/figures
./MDIQA/MDIQA
./MDIQA/MDIQA/models
./MDIQA/MDIQA/archs
./MDIQA/MDIQA/losses
./MDIQA/MDIQA/options_MDIQA
./MDIQA/MDIQA/utils
./MDIQA/MDIQA/data
./MDIQA/pyiqa.egg-info
./MDIQA/scripts
./MDIQA/BasicSR
./MDIQA/BasicSR/docs
./MDIQA/BasicSR/options
./MDIQA/BasicSR/basicsr
./MDIQA/BasicSR/assets
./MDIQA/BasicSR/test_scripts
./MDIQA/BasicSR/basicsr.egg-info
./MDIQA/BasicSR/inference
./MDIQA/BasicSR/scripts
./MDIQA/BasicSR/tests
./MDIQA/BasicSR/colab
./MDIQA/BasicSR/LICENSE
./MDIQA/tests
./MDIQA/tests/test_efficiency_img_dir
./MDIQA/open_clip
./MDIQA/open_clip/docs
./MDIQA/open_clip/src
./MDIQA/open_clip/tutorials
./MDIQA/open_clip/.github
./MDIQA/open_clip/scripts
./MDIQA/open_clip/tests
./android_gemma_latency_harness
./android_gemma_latency_harness/app
./android_gemma_latency_harness/app/src
./scripts
./scripts/__pycache__
./scripts/technical_iqa
./scripts/technical_iqa/__pycache__
./test_outputs
./test_outputs/stdout_validation
./test_outputs/mobile_eval
./test_outputs/distill_dry_run
./test_outputs/vila_selector_smoke
./test_outputs/vila_selector_smoke/output
./test_outputs/diversity_smoke
./test_outputs/diversity_smoke/off
./test_outputs/diversity_smoke/on
./test_outputs/real_image_validation
./test_outputs/validation_no_api_key
./test_outputs/validation_with_api_key
./test_outputs/distill_dry_run_probe
./test_outputs/vila_hf_clip_fix_validation
./test_outputs/stdout_validation_2
./test_samples
./experiments
./experiments/technical
./experiments/fusion
./experiments/icaa_export_safe_v3
./experiments/icaa_export_safe_v3/reports
./experiments/icaa_export_safe_v3/export_safe_models
./experiments/icaa_export_safe_v3/scripts
./experiments/aesthetic
./experiments/icaa_tf_native
./experiments/icaa_tf_native/reports
./experiments/icaa_tf_native/weight_maps
./experiments/icaa_tf_native/scripts
./experiments/icaa_tf_native/tf_models
./experiments/icaa_export_safe
./experiments/icaa_export_safe/reports
./experiments/icaa_export_safe/export_safe_models
./experiments/icaa_export_safe/scripts
./experiments/ranking
./experiments/composition
./experiments/icaa_export_safe_v2
./experiments/icaa_export_safe_v2/reports
./experiments/icaa_export_safe_v2/export_safe_models
./experiments/icaa_export_safe_v2/scripts
./data
./data/debug
./data/debug/c6_parity_20260527
./data/raw
./data/raw/flive
./data/raw/paq2piq
./data/raw/.codex
./data/raw/.git
./data/raw/private_mobile
./data/raw/koniq10k
./data/raw/aadb
./data/raw/icaa17k
./data/raw/.agents
./data/raw/spaq
./data/Test_weight
./data/Test_weight/Soso
./data/Test_weight/Bad
./data/Test_weight/Good
./data/processed
./data/processed/distortionguard_iqa_v1
./data/processed/flive
./data/processed/ava
./data/processed/paq2piq
./data/processed/ava_captions
./data/processed/private_mobile
./data/processed/koniq10k
./data/processed/techiqa_guard
./data/processed/aadb
./data/processed/ranking
./data/processed/topiq_replacement
./data/processed/spaq
./data/processed/smoke
./data/technical_distill
./data/technical_distill/manifests
./data/technical_distill/teacher_labels
./data/TW
./.agents
./.agents/skills
./.agents/skills/to-issues
./.agents/skills/grill-me
./.agents/skills/improve-codebase-architecture
./.agents/skills/handoff
./.agents/skills/to-prd
./.agents/skills/grill-with-docs
./open_clip
./open_clip/docs
./open_clip/docs/script_examples
./open_clip/src
./open_clip/src/open_clip_train
./open_clip/src/open_clip
./open_clip/.git
./open_clip/.git/info
./open_clip/.git/refs
./open_clip/.git/logs
./open_clip/.git/objects
./open_clip/.git/branches
./open_clip/.git/hooks
./open_clip/tutorials
./open_clip/.pdm-build
./open_clip/.github
./open_clip/.github/workflows
./open_clip/scripts
./open_clip/tests
./exports
./exports/tflite
./exports/tflite/musiq_koniq_probe1024
./exports/tflite/musiq_flive_probe1024
./exports/tflite/musiq_spaq_probe1024
```

## Existing output directories

```text
outputs/mdiqa_flive_protocol_audit_20260601
outputs/topiq_lite_layer_check
outputs/rgnet_float32_retry_20260508
outputs/alamp_paper_fusion_4096_graphlite_retry_v1
outputs/aesthetic_ensemble_production_precheck_20260603
outputs/firebase_worker_local_smoke
outputs/mobile_student_iqa_v4a_20260601_smoke_fixed
outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221456
outputs/icaa_export_safe_onnx_20260514_210121
outputs/test_vila_nvila_resume_20260415
outputs/icaa_tf_native_savedmodel_20260516_125422
outputs/probe_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_131131
outputs/icaa_tf_native_savedmodel_20260516_044158
outputs/distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523
outputs/dryrun_distortionguard_stageB2_partial_unfreeze_b3_20260524
outputs/alamp_object_graph_subset_20260511
outputs/icaa_tf_native_tflite_fp32_20260516_132108
outputs/icaa_tf_native_tflite_fp32_20260516_140027
outputs/nima_fixed_preproc_eval_20260525
outputs/hybrid_teacher_adjusted_calibration_2k_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_125633
outputs/saved_models
outputs/distortionguard_stageA_synthetic_medium500_unfrozen_continue_e10_20260523
outputs/icaa_export_safe_onnx_20260514_194704
outputs/alamp_paper_faithfulness_audit_20260511
outputs/acut_stage4_nima_musiq_alamp_rgnet
outputs/acut_icaa_skip_weight_search_20260601
outputs/smallrun_alamp_multipatch_teacher_20260524
outputs/icaa_tf_native_tflite_fp32_20260516_130721
outputs/icaa_export_safe_onnx_20260514_194438
outputs/icaa_tf_native_savedmodel_20260516_132933
outputs/icaa_tf_native_savedmodel_20260516_130941
outputs/acut_product_vila_local_real
outputs/eval_topiq_lite_ranking_probe_lam02_gap10_e10_20260517
outputs/hybrid_teacher_unknown_fallback_audit_20260603
outputs/probe1024_topiq_lite_flive_e15_frozen
outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528
outputs/topiq_teacher_label_expansion_50k_20260601
outputs/distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524
outputs/alamp_restart
outputs/distill_exec_20260411
outputs/topiq_flive_heuristic_gate_full_simulation_20260608
outputs/alamp_paper_fusion_1024_graphlite_20260513
outputs/rgnet_old_vs_paper_fp16_compare_20260514
outputs/icaa_tf_native_savedmodel_20260516_125945
outputs/icaa_tf_native_stage_f_20260516_113010
outputs/icaa_tf_native_savedmodel_20260516_042006
outputs/icaa_tf_native_tflite_fp32_20260516_120948
outputs/acut_product_gold_weight_eval_20260603
outputs/technical_iqa_signal_comparison_20260604
outputs/icaa_tf_native_tflite_fp32_20260516_132825
outputs/ablate_vila_010
outputs/acut_product_ready_smoke_post_integration
outputs/pairwise_restart
outputs/icaa_export_safe_v3_pytorch_parity_20260515_170845
outputs/icaa_int8_qat_audit_20260601
outputs/final_topiq_lite_mixed112_export_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_043548
outputs/smoke_topiq_lite_spaq
outputs/icaa_tf_native_stage_g3_20260516_113510
outputs/acut_weekly_research_journal_20260526
outputs/full_iqa_iaa_model_design_audit_20260611
outputs/icaa_tf_native_tflite_fp32_20260516_121121
outputs/ablate_vila_015
outputs/acut_product_vila_local_score_reasoning_v10_probe
outputs/vila_scores_run
outputs/rgnet_paper_aadb_tflite_export_20260514
outputs/icaa_tf_native_stage_c_20260515_175817
outputs/alamp_v4_full_ava_20260517
outputs/eval_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517
outputs/icaa_export_safe_v2_onnx_20260514_215455
outputs/firebase_worker_new_path_verify_gpu
outputs/smallrun_alamp_multipatch_teacher_classweighted_block5_20260524
outputs/eval_topiq_lite_ranking_probe_lam05_e12_20260517
outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525
outputs/gemini_api_teacher_aesthetic3_20260427_revised
outputs/icaa_torchscript_parity_20260514_151359
outputs/test_vila_aesthetic_direct_compare_20260504_153044
outputs/distortionguard_stageB3_flive_weighted_diagnosis_20260524
outputs/acut_product_nvila8b_probe_v5
outputs/research_protocol_bounded_eval_20260530
outputs/icaa_performance_reverification_20260525
outputs/smoke_distortionguard_stageA_synthetic_medium500_e2_20260523
outputs/full_topiq_lite_spaq_frozen_e30_20260517
outputs/topiq_teacher_guard_design_20260531
outputs/icaa_tf_native_savedmodel_20260516_134439
outputs/firebase_worker_normal_path_verify
outputs/acut_product_vila_local_score_reasoning_v3
outputs/icaa_tf_native_tflite_fp32_20260516_122927
outputs/eval_tflite_technical_compare
outputs/icaa_tf_native_savedmodel_20260516_124619
outputs/icaa_tf_native_tflite_fp32_20260516_131805
outputs/icaa_tf_native_tflite_fp32_20260516_132937
outputs/rgnet_v1_aadb_pil_resize_retrain_20260524_tflite
outputs/dryrun_rgnet_v1_cascaded_denseaspp_aadb_20260525
outputs/icaa_tf_native_stage_g1_20260516_113112
outputs/alamp_paper_mpnet_design_audit_20260511
outputs/ranking_topiq_lite_mixed112_flive_pairs_lam02_gap10_e10_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_132805
outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609
outputs/probe_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517
outputs/icaa_tf_native_stage_b_20260515_174854
outputs/icaa_tf_native_savedmodel_20260516_124135
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526
outputs/icaa_tf_native_tflite_fp32_20260516_124410
outputs/capstone_model_audit_20260711
outputs/acut_product_ready_diversity_check
outputs/dryrun_alamp_multipatch_teacher_sgd_default_compat_20260525
outputs/acut_product_ready_smoke
outputs/ranking_topiq_lite_mixed112_flive_pairs_lam05_e3_20260517
outputs/acut_aesthetic_benchmark_csv_audit_20260603
outputs/paired_production_mdiqa_gpu_bottleneck_audit_20260603
outputs/icaa_official_eval_20260514_141732
outputs/icaa_tf_native_savedmodel_20260516_123508
outputs/icaa_tf_native_stage_g2_mlp_debug_20260516_031415
outputs/acut_capstone_final_report_20260526
outputs/aesthetic_ensemble_weight_search_20260603
outputs/probe1024_topiq_lite_spaq_e15_unfrozen
outputs/rgnet_restart
outputs/icaa_tf_native_tflite_fp32_20260516_132453
outputs/mobile_alamp_v2_pretrained_4096
outputs/topiq_mixed112_artifact_recovery_20260527
outputs/rgnet_v1_aadb_pil_resize_retrain_20260524
outputs/icaa_tf_native_stage_f_20260515_183424
outputs/dryrun_techiqa_guard_v1_stage4_fpdata_check_20260520
outputs/aesthetic_cross_predictions_for_weight_search_20260603
outputs/firebase_worker_local_smoke_post_integration
outputs/acut_product_vila_local_format_check_v4
outputs/mobile_student_iqa_v4b_checkpoint_reeval_20260601
outputs/icaa_tf_native_savedmodel_20260516_133929
outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517
outputs/distortionguard_hard_fp_v2_failure_taxonomy_20260524
outputs/topiq_flive_safety_gate_simulation_20260608
outputs/alamp_paper_fusion_smoke_20260513_fix
outputs/acut_product_contract_check
outputs/mobile_student_iqa_14105_smoke_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_121312
outputs/icaa_tf_native_tflite_fp32_20260516_121625
outputs/icaa_tf_native_savedmodel_20260516_121445
outputs/mobile_alamp_v2_4096_smoke
outputs/final_topiq_lite_ranking_lam01_export_20260517
outputs/alamp_multipatch_teacher_classweighted_block5_full_20260524
outputs/icaa_tf_native_savedmodel_20260516_123244
outputs/student_flive_hybrid_benchmark_20260604
outputs/mdiqa_variant_sanity_20260601
outputs/test_vila_raw_aesthetic_scores_html_20260424
outputs/mobile_student_iqa_v2_error_analysis_20260601
outputs/icaa_tf_native_stage_h_full_dat_20260516_040004
outputs/eval_distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524
outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609
outputs/icaa_tf_native_tflite_fp32_20260516_130846
outputs/official_protocol_benchmark_manifest_20260604
outputs/hybrid_teacher_label_dryrun_10k_20260603
outputs/alamp_gpu_acceleration_audit_20260603
outputs/firebase_worker_stage5_student_nvila_fix_verify
outputs/smoke_topiq_rankcalib_v2_stage1_e3_20260517
outputs/icaa_ptq_candidates_20260601
outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427
outputs/acut_product_contract_check_smoke_diversity
outputs/icaa_export_safe_v3_onnx_20260515_171049
outputs/dryrun_alamp_multipatch_teacher_papercap4096_sgd_20260525
outputs/smallrun_alamp_dual_branch_teacher_4096_20260524
outputs/icaa_export_safe_original_parity_20260514_153136
outputs/gemma4_mobile_export_probe_20260429
outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525
outputs/icaa_tf_native_tflite_fp32_20260516_130121
outputs/stage5_runs
outputs/icaa_tf_native_tflite_fp32_20260516_134314
outputs/mobile_student_iqa_50k_smoke_20260608
outputs/topiq_flive_safety_signal_ablation_20260608
outputs/paper_protocol_technical_iqa_audit_20260530
outputs/icaa_tf_native_tflite_fp32_20260516_131903
outputs/techiqa_guard_v1_direct_stage5_rank002_fpguard005_20260520
outputs/mdiqa_official_protocol_expansion_audit_20260603
outputs/acut_product_nvila8b_probe
outputs/icaa_tf_native_tflite_fp32_20260516_125317
outputs/eval_alamp_multipatch_teacher_classweighted_block5_full_20260524
outputs/musiq_kaggle_smoke_20260601
outputs/alamp_paper_fusion_4096_chunked_graphlite
outputs/ablate_vila_refined_010
outputs/icaa_tf_native_tflite_fp16_20260516_143721
outputs/icaa_tf_native_tflite_fp32_20260516_130352
outputs/icaa_tf_native_tflite_fp32_20260516_043326
outputs/smallrun_alamp_multipatch_teacher_papercap4096_20260525
outputs/icaa_export_torchscript_20260514_150436
outputs/alamp_paper_fusion_4096_graphlite_bs2
outputs/dryrun_rgnet_v1_hybrid_spatial_20260525
outputs/acut_product_ready_smoke_contract_wrapper_skip
outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609
outputs/icaa_tf_native_savedmodel_20260516_123326
outputs/ablate_vila_refined_005
outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609
outputs/ranking_pairgen_probe_gap010_tmp_20260517
outputs/technical_iqa_combo_benchmark_20260524
outputs/mobile_student_iqa_14105_tflite_export_20260604
outputs/icaa_tf_native_tflite_fp32_20260516_132922
outputs/alamp_rgnet_aadb_ava_comparison_20260513
outputs/icaa_tf_native_tflite_fp32_20260516_132545
outputs/icaa_tf_native_savedmodel_20260516_125914
outputs/probe1024_topiq_lite_spaq_e15_frozen
outputs/alamp_paper_ava_v1_design_audit_20260511
outputs/icaa_tf_native_stage_e_20260515_181850
outputs/research_protocol_image_test_eval_20260530
outputs/icaa_tf_native_stage_d_20260515_180922
outputs/dryrun_alamp_dual_branch_teacher_20260524
outputs/dryrun_alamp_multipatch_teacher_recipe_default_compat_20260525
outputs/mobile_student_iqa_50k_controlled_b0_20260608
outputs/eval_final_topiq_candidates_vs_existing_technical_20260520
outputs/logs
outputs/acut_model_core_training_code_excerpt_20260601
outputs/icaa_tf_native_savedmodel_20260516_130526
outputs/student_training_manifest_design_20260601
outputs/test_vila_aesthetic_model_audit_20260423
outputs/eval_topiq_rankcalib_v2_stage1_regression_e15_20260517
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_smoke_fixed_20260517
outputs/rgnet_model_lineage_audit_20260513
outputs/eval_full_topiq_lite_spaq_replacement_20260517
outputs/alamp_patch_fidelity_audit_20260525
outputs/icaa_tf_native_stage_h_full_dat_20260516_040212
outputs/hybrid_teacher_range_calibration_2k_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_132647
outputs/color_aesthetics_audit_20260509
outputs/icaa_tf_native_tflite_fp32_20260516_122056
outputs/mdiqa_smoke_20260601
outputs/icaa_tf_native_savedmodel_20260516_123925
outputs/probe1024_topiq_lite_flive_e15_unfrozen
outputs/acut_final_ratio_core_eval_20260603
outputs/probe1024_musiq_flive
outputs/alamp_dual_branch_gcn_failure_analysis_20260524
outputs/mobile_student_iqa_v4b_20260601_sanity
outputs/acut_stage5_full_with_pairwise
outputs/alamp_paper_fusion_4096_graphlite_oom_partial_20260514_1339
outputs/dryrun_alamp_multipatch_teacher_20260524
outputs/acut_product_vila_local_probe
outputs/official_protocol_mdiqa_signal_benchmark_20260604
outputs/dryrun_rgnet_v1_cascaded_denseaspp_paper_recipe_20260526
outputs/stage5_with_vila_reasoning
outputs/icaa_tf_native_stage_g2_20260516_024326
outputs/icaa_tf_native_tflite_fp32_20260516_134508
outputs/firebase_worker_new_path_verify
outputs/known_dataset_full_label_dryrun_20260603
outputs/icaa_tf_native_savedmodel_20260516_121053
outputs/eval_topiq_lite_mixed112_unfreeze_replacement_20260517
outputs/c6_android_wsl_parity_exif_fixed_20260528
outputs/dryrun_rgnet_v1_aadb_pil_resize_retrain_20260524
outputs/eval_topiq_lite_flive_calibration_replacement_20260517
outputs/acut_product_vila_local_score_reasoning_v9
outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608
outputs/dryrun_alamp_multipatch_teacher_classweighted_block5_20260524
outputs/icaa_tf_native_tflite_fp32_20260516_114655
outputs/research_protocol_image_test_eval_analysis_20260530
outputs/eval_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
outputs/icaa_tf_native_stage_h_full_dat_20260516_040045
outputs/icaa_tf_native_tflite_fp32_20260516_130645
outputs/acut_product_vila_local_score_reasoning_v6
outputs/alamp_paper_fusion_smoke_20260513_finalguard
outputs/midrun_alamp_multipatch_teacher_papercap4096_8192_20260525
outputs/mobile_alamp_v2_pretrained_smoke
outputs/aesthetic_mobile_paper_benchmark_20260603
outputs/mobile_alamp_v2_pretrained_probe_1024
outputs/alamp_detector_selection_20260511
outputs/acut_product_nvila8b_probe_v10_gpu
outputs/icaa_tf_native_tflite_fp32_20260516_132741
outputs/rgnet_paper_aadb_mobile_export_audit_20260513
outputs/musiq_restart
outputs/nima_underperformance_diagnosis_20260525
outputs/icaa_tf_native_savedmodel_20260516_125453
outputs/research_protocol_smoke_scoring_20260530
outputs/icaa_torchscript_parity_20260514_151034
outputs/techiqa_guard_v1_direct_stage2_partial_unfreeze_b3_flive15_20260520
outputs/icaa_tf_native_savedmodel_20260516_125309
outputs/ava_pretrain_aadb_finetune_20260507
outputs/alamp_external_patch_validation_20260524
outputs/probe1024_topiq_lite_koniq_e15_unfrozen
outputs/icaa_tf_native_savedmodel_20260516_130441
outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526
outputs/icaa_export_safe_torchscript_20260514_195503
outputs/alamp_paper_mpnet_patch_selector_v2_20260511
outputs/alamp_paper_ava_v2_graph_lite_20260511
outputs/acut_product_vila_local_score_reasoning_v4
outputs/ava_retrain_completion_audit_20260507
outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549
outputs/icaa_tf_native_stage_f_20260515_182930
outputs/dryrun_rgnet_v1_default_context_compat_20260525
outputs/mixed_replay_topiq_lite_112_unfrozen_lr1e6_e12_20260517
outputs/eval_techiqa_guard_v1_stage4b_fpconfirmed_v2_lam005_20260522
outputs/icaa_tf_native_savedmodel_20260516_130030
outputs/mdiqa_cuda_diagnosis_20260601
outputs/gemini_api_teacher_aesthetic3_20260427
outputs/icaa_torchscript_parity_20260514_151144
outputs/acut_product_vila_local_score_reasoning_v2
outputs/mdiqa_checkpoint_source_audit_20260601
outputs/mdiqa_student_final_no_go_report_20260611
outputs/mobile_student_iqa_v3_error_analysis_20260601
outputs/dryrun_alamp_multipatch_teacher_papercap4096_recipe_20260525
outputs/nima_ava_restart
outputs/icaa_tf_native_tflite_fp32_20260516_132629
outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526
outputs/techiqa_guard_v1_direct_stage5_rank001_fpguard05_20260520
outputs/icaa_tf_native_tflite_fp32_20260516_042524
outputs/acut_transparency_quick_export_check
outputs/stage5_selector_only
outputs/icaa_tf_native_stage_h_full_dat_20260516_113539
outputs/icaa_tf_native_stage_f_20260516_113458
outputs/stage5_with_vila_reasoning_refined
outputs/icaa_export_safe_v2_pytorch_parity_20260514_215433
outputs/mobile_student_iqa_v4b_20260601_smoke_fixed
outputs/musiq_kaggle_sanity_fallback_20260601
outputs/ablate_vila_refined_000
outputs/c6_topiq_tensor_source_dimension_match_20260529
outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke
outputs/alamp_paper_fusion_smoke_20260513
outputs/acut_product_contract_check_diversity
outputs/ablate_vila_000
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full
outputs/icaa_tf_native_stage_g3_20260516_113200
outputs/alamp_layout_aware_teacher_plan_20260524
outputs/icaa_tf_native_stage_f_20260515_183750
outputs/icaa_tf_native_savedmodel_20260516_125059
outputs/mpnet_gpu_env_audit_20260512
outputs/research_protocol_bounded_eval_analysis_20260530
outputs/full_topiq_lite_unified_to_flive_calib_frozen_e20_20260517
outputs/dryrun_techiqa_guard_v1_stage2_partial_unfreeze_20260520
outputs/alamp_paper_strict_patch_selector_20260512
outputs/topiq_lite_replacement_export_feasibility_20260517
outputs/eval_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517
outputs/midrun_alamp_multipatch_teacher_papercap4096_sgd1e4_8192_20260525
outputs/midrun_alamp_multipatch_teacher_papercap4096_sgd1e3_8192_20260525
outputs/icaa_tflite_conversion_20260514_200500
outputs/probe_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
outputs/eval_topiq_unified_full_flive_replacement_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_132044
outputs/alamp_paper_mpnet_patch_selector_v4_20260512
outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke
outputs/hybrid_teacher_label_design_review_2k_20260603
outputs/acut_product_vila_local_score_reasoning_v11_probe
outputs/alamp_dual_branch_gcn_teacher_preflight_20260524
outputs/topiq_teacher_performance_audit_20260601
outputs/eval_techiqa_guard_v1_stage5_rank001_fpguard05_20260520
outputs/icaa_tf_native_tflite_fp32_20260516_134634
outputs/paper_level_stage1_benchmark_alignment_20260523
outputs/mdiqa_correct_protocol_full_benchmark_20260601
outputs/icaa_tf_native_savedmodel_20260516_123646
outputs/eval_alamp_multipatch_teacher_full_ava_20260524_smoke
outputs/probe1024_topiq_lite_koniq_e15_frozen
outputs/technical_iqa_signal_comparison_with_c6_20260604
outputs/icaa_tf_native_tflite_fp32_20260516_132223
outputs/alamp_mpnet_vs_aadb_alamp_review_20260513
outputs/final_topiq_mixed112_replacement_decision_20260520
outputs/dryrun_rgnet_v1_default_recipe_compat_20260526
outputs/c6_android_wsl_parity_exif_fixed_20260529
outputs/distortionguard_stageA_synthetic_medium500_unfrozen_e5_20260523
outputs/alamp_external_patch_full_conversion_20260524
outputs/icaa_tf_native_tflite_fp32_20260516_115302
outputs/rgnet_paper_aadb_regression_20260509
outputs/test_vila_aesthetic_direct_compare_20260504_135316
outputs/mobile_student_iqa_v4b_20260601_smoke
outputs/hybrid_teacher_label_design_review_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_131503
outputs/icaa_tf_native_tflite_fp32_20260516_121509
outputs/mdiqa_student_gap_analysis_20260608
outputs/capstone_technical_iqa_baseline_summary_20260530
outputs/icaa_tf_native_savedmodel_20260516_043231
outputs/icaa_tf_native_savedmodel_20260516_115347
outputs/rgnet_paper_faithfulness_next_step_20260525
outputs/student_conservative_aesthetic_full_aadb_20260423
outputs/acut_product_backend_cleanup_smoke
outputs/icaa_tf_native_savedmodel_20260516_122031
outputs/eval_topiq_lite_vs_mobile_baselines_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_123037
outputs/topiq_teacher_smoke_20260531
outputs/nima_ava_full
outputs/icaa_tf_native_tflite_fp32_20260516_130947
outputs/acut_stage5_full_with_pairwise_fast
outputs/icaa_tf_native_tflite_fp32_20260516_132243
outputs/techiqa_guard_v1_direct_stage4b_fpconfirmed_v2_lam005_20260522
outputs/icaa_tf_native_savedmodel_20260516_130840
outputs/icaa_tf_native_tflite_fp32_20260516_123819
outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523
outputs/mdiqa_student_pre_final_audit_20260609
outputs/icaa_qat_attempt_20260601
outputs/thesis_material_audit_20260607
outputs/audit_model_paper_basis_20260506
outputs/alamp_paper_mpnet_selector_compare_20260512
outputs/known_dataset_full_label_coverage_audit_20260603
outputs/acut_product_vila_local_format_check
outputs/icaa_tf_native_stage_g0_20260516_014248
outputs/icaa_export_safe_torchscript_20260514_153437
outputs/icaa_tf_native_savedmodel_20260516_130155
outputs/eval_topiq_lite_spaq_to_koniq_replacement_20260517
outputs/techiqa_guard_v1_direct_stage4_fpdata_lam005_20260520
outputs/icaa_tf_native_tflite_fp32_20260516_132720
outputs/mobile_student_iqa_14105_checkpoint_eval_20260604
outputs/icaa_tf_native_savedmodel_20260516_050514
outputs/smoke_topiq_rankcalib_v2_stage3_distill_lam1_rank002_e3_20260517
outputs/icaa_tf_native_savedmodel_20260516_123605
outputs/icaa_export_safe_onnx_20260514_210640
outputs/test_vila_aesthetic_ensemble_experiment_20260423
outputs/alamp_external_repo_audit_20260511
outputs/distill_conservative_aesthetic_full_aadb_20260423
outputs/acut_product_nvila8b_probe_v7_gpu
outputs/student_training_manifest_50k_20260601
outputs/smoke_techiqa_guard_v1_direct_e3_20260520
outputs/rgnet_paper_v1_ablation_20260509
outputs/tech_iqa_guard_v1_dataset_design_20260520
outputs/icaa_export_safe_full_eval_20260514_193609
outputs/mobile_student_iqa_v3_20260601_smoke
outputs/alamp_multipatch_teacher_improvement_plan_20260524
outputs/test_vila_aesthetic_direct_compare_20260504_153732
outputs/probe1024_musiq_koniq
outputs/c6_android_parity_final_audit_20260530
outputs/mobile_student_iqa_14105_smoke_gpu_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_132021
outputs/rgnet_v1_cascaded_denseaspp_aadb_full_20260525
outputs/mobile_student_iqa_v1_20260601_smoke
outputs/acut_product_ready_smoke_contract
outputs/acut_product_gold_weight_eval_test_vila_20260603
outputs/acut_product_vila_local_format_check_v3
outputs/gemini_api_teacher_aesthetic3_highspec_100_batch_20260427
outputs/stage5_comparisons
outputs/alamp_paper_mpnet_patch_selector_v3_20260512
outputs/research_protocol_production_technical_eval_analysis_20260530
outputs/topiq_student_distill
outputs/mobile_student_iqa_v2_20260601_smoke
outputs/dryrun_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520
outputs/acut_product_vila_local_format_check_v7
outputs/icaa_official_eval_20260514_141838
outputs/paper_faithful_audit_20260509
outputs/icaa_tf_native_tflite_fp32_20260516_125534
outputs/icaa_tf_native_stage_g1_20260516_022555
outputs/mobile_alamp_v2_full_ava_20260519
outputs/alamp_multipatch_teacher_full_ava_20260524
outputs/smoke_musiq_koniq
outputs/alamp_patch_overlay_audit_20260513
outputs/final_c6_log_only_readiness_20260528
outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608
outputs/icaa_export_safe_original_parity_20260514_153239
outputs/acut_product_vila_local_format_check_v5
outputs/icaa_tf_native_savedmodel_20260516_121244
outputs/alamp_paper_fusion_4096_chunked_graphlite_smoke
outputs/icaa_tf_native_stage_f_20260516_113327
outputs/techiqa_guard_v1_direct_stage3_fpguard_lam005_20260520
outputs/topiq_teacher_guard_holdout_20260601
outputs/eval_rgnet_v1_cascaded_denseaspp_aadb_full_20260525
outputs/techiqa_guard_v1_hard_fp_mining_20260522
outputs/icaa_export_safe_original_parity_20260514_185821
outputs/acut_product_nvila8b_probe_v2
outputs/test_vila_aesthetic_direct_compare_20260504_153222
outputs/ablate_vila_refined_015
outputs/icaa_tf_native_tflite_fp32_20260516_131223
outputs/icaa_tf_native_savedmodel_20260516_130549
outputs/test_vila_worker_verify_full_20260416
outputs/acut_product_gemini_backend_sample
outputs/icaa_export_safe_onnx_20260514_195608
outputs/aesthetic_mobile_paper_benchmark_20260603_audit
outputs/icaa_tf_native_tflite_fp32_20260516_130042
outputs/stage5_with_vila_reasoning_smoke_refined
outputs/smoke_distortionguard_stageB_authentic_from_stageA_e3_20260523
outputs/eval_techiqa_guard_v1_stage2_partial_unfreeze_b3_flive15_20260520
outputs/icaa_tf_native_tflite_fp32_20260516_042626
outputs/dryrun_distortionguard_stageA_init_weights_20260523
outputs/icaa_tf_native_savedmodel_20260516_123215
outputs/icaa_tf_native_savedmodel_20260516_122852
outputs/mdiqa_distillation_50k_manifest_audit_20260608
outputs/mixed_replay_topiq_lite_111_frozen_e15_20260517
outputs/dryrun_rgnet_ava_repro_full_1ep_20260526
outputs/ava_retrain_rgnet_alamp_20260506
outputs/thesis_material_final_correction_20260607
outputs/paq2piq_official_split_search_20260530
outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221739
outputs/icaa_tf_native_savedmodel_20260516_125026
outputs/distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524
outputs/alamp_papercap_recipe_audit_20260525
outputs/ablate_vila_005
outputs/alamp_external_integration_feasibility_audit_20260524
outputs/mpnet_gpu_cudnn_audit_20260513
outputs/alamp_dual_branch_4096_comparison_20260524
outputs/official_protocol_technical_signal_benchmark_20260604
outputs/icaa_tf_native_savedmodel_20260516_125701
outputs/mdiqa_distillation_50k_full_targets_20260608
outputs/mobile_alamp_v2_full_ava_20260519_tflite
outputs/icaa_tf_native_savedmodel_20260516_130422
outputs/acut_product_ready_smoke_diversity_post_integration
outputs/rgnet_paper_v1_ablation_full_candidates_20260510
outputs/mobile_student_iqa_v2_20260601_fixed
outputs/icaa_tf_native_stage_g0_20260516_113059
outputs/model_strategy_review_20260506
outputs/eval_techiqa_guard_v1_stage3_fpguard_lam005_20260520
outputs/acut_product_nvila8b_probe_v8_gpu
outputs/icaa_tf_native_tflite_fp32_20260516_134032
outputs/eval_techiqa_guard_v1_stage5_rank002_fpguard005_v2_20260520
outputs/icaa_tf_native_savedmodel_20260516_113654
outputs/alamp_paper_fusion_1024_comparison_20260513
outputs/final_c6_topiq_koniq_technical_score_decision_20260527
outputs/acut_stage3_nima_musiq_alamp
outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604
outputs/mobile_student_iqa_v1_20260601
outputs/mobile_student_iqa_v4a_20260601_smoke
outputs/final_techiqa_guard_v1_nogo_20260522
outputs/icaa_tf_native_savedmodel_20260516_121604
outputs/probe1024_musiq_spaq
outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609
outputs/icaa_export_safe_onnx_20260514_194833
outputs/paper_level_aesthetic_gap_audit_20260523
outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist03_rank0_e3_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_132300
outputs/icaa_tf_native_stage_g3_dattention_debug_20260516_034800
outputs/icaa_tf_native_stage_f_20260515_183945
outputs/ranking_topiq_lite_mixed112_flive_pairs_lam05_e12_20260517
outputs/paired_production_mdiqa_significance_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_121933
outputs/acut_product_firebase_regression
outputs/acut_product_vila_local_score_reasoning_v10
outputs/aesthetic_weight_lab_structure_audit_20260508
outputs/mixed_replay_topiq_lite_221_frozen_e15_20260517
outputs/dryrun_distortionguard_stageB3_flive_weighted_20260524
outputs/eval_techiqa_guard_v1_stage5_rank002_fpguard005_20260520
outputs/koniq_android_wsl_parity_variant_diagnosis_20260528
outputs/mdiqa_sanity_20260601
outputs/rgnet_mobile_resize_finetune_preflight_20260524
outputs/icaa_tf_native_savedmodel_20260516_130104
outputs/mdiqa_wsl_feasibility_audit_20260601
outputs/icaa_export_safe_onnx_20260514_211156
outputs/acut_product_nvila8b_probe_v3
outputs/icaa_tf_native_tflite_fp32_20260516_124204
outputs/nima_icaa_paper_level_eval_20260525
outputs/icaa_tf_native_stage_g3_20260516_033136
outputs/mobile_alamp_v2_4096_gpu_smoke
outputs/icaa_tf_native_tflite_fp32_20260516_132136
outputs/icaa_tf_native_tflite_fp32_20260516_125153
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526
outputs/rgnet_faithful_architecture_audit_20260524
outputs/icaa_tf_native_savedmodel_20260516_134232
outputs/icaa_tf_native_tflite_fp32_20260516_132524
outputs/icaa_export_safe_onnx_20260514_210025
outputs/alamp_paper_fusion_1024_graphgcn_20260513
outputs/icaa_tf_native_state_dict_inspection
outputs/icaa_export_safe_torchscript_20260514_195550
outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517
outputs/final_research_protocol_technical_iqa_baseline_report_20260530
outputs/distill_gpu_runtime_check
outputs/test_vila_aesthetic_only_stage5_current_conservative_20260423
outputs/mobile_student_iqa_v2_training_loop_verify_20260601
outputs/research_protocol_iqa_manifests_20260530
outputs/acut_product_vila_local_score_reasoning_v8
outputs/acut_product_ready_smoke_contract_wrapper
outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526
outputs/firebase_worker_local_smoke_diversity
outputs/eval_techiqa_guard_v1_stage4_fpdata_lam005_20260520
outputs/icaa_tf_native_tflite_fp32_20260516_125926
outputs/icaa_tf_native_savedmodel_20260516_122748
outputs/icaa_tf_native_tflite_fp32_20260516_130316
outputs/icaa_export_safe_original_parity_20260514_153215
outputs/icaa_tf_native_savedmodel_20260516_044408
outputs/acut_weekly_research_journal_v2_20260526
outputs/musiq_kaggle_sanity_20260601
outputs/gemma4_e4b_aesthetic3_expanded_20260427
outputs/acut_model_core_code_audit_20260601
outputs/technical_teacher_hybrid_design_audit_20260603
outputs/dryrun_rgnet_v1_hybrid_spatial_default_compat_20260525
outputs/alamp_paper_mpnet_4096_expansion
outputs/acut_product_vila_local_format_check_v6
outputs/rgnet_paper_ava_classification_20260510
outputs/eval_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517
outputs/icaa_export_safe_v3_tflite_probe_20260515_171113
outputs/acut_product_vila_local_score_reasoning_v7
outputs/icaa_tf_native_savedmodel_20260516_130340
outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526
outputs/mobile_alamp_v2_1024_probe
outputs/acut_product_nvila8b_probe_v7
outputs/topiq_lite_batch_check
outputs/smoke_musiq_flive
outputs/paq2piq_external_official_split_search_20260530
outputs/smoke_topiq_rankcalib_v2_stage2_flive_rank_lam005_e3_20260517
outputs/midrun_alamp_multipatch_teacher_papercap4096_recipe_8192_20260525
outputs/acut_product_nvila8b_probe_v9_gpu
outputs/mobile_student_iqa_v4b_20260601
outputs/icaa_tf_native_savedmodel_20260516_125005
outputs/icaa_tf_native_savedmodel_20260516_121906
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517
outputs/acut_product_vila_local_score_reasoning_v5
outputs/acut_product_vila_local_real2
outputs/musiq_kaggle_failure_diagnosis_20260601
outputs/rgnet_paper_v1_aadb_regression_20260509
outputs/mobile_student_iqa_v4a_20260601
outputs/eval_topiq_lite_mixed_replay_replacement_20260517
outputs/eval_distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524
outputs/mdiqa_official_variant_sanity_20260601
outputs/musiq_wsl_kaggle_audit_20260601
outputs/icaa_export_safe_onnx_20260514_195654
outputs/c6_android_wsl_parity_20260527
outputs/smoke_techiqa_guard_v1_build
outputs/mobile_student_iqa_14105_short_train_20260604
outputs/techiqa_guard_v1_direct_stage1_frozen_20260520
outputs/eval_alamp_multipatch_teacher_full_ava_20260524
outputs/distortionguard_iqa_v1_design_20260523
outputs/test_vila_human_acut_review_stage5_vs_ensemble_20260423
outputs/icaa_export_safe_original_parity_20260514_195312
outputs/topiq_teacher_validation_20260531
outputs/alamp_salient_detector_audit_20260512
outputs/icaa_export_safe_original_parity_20260514_153053
outputs/icaa_tf_native_savedmodel_20260516_130639
outputs/paper_level_teacher_reproduction_plan_20260524
outputs/thesis_material_correction_audit_20260607
outputs/aesthetic_ensemble_weight_search_complete_20260603_audit
outputs/icaa_export_safe_onnx_20260514_202941
outputs/acut_product_nvila8b_probe_v6
outputs/mobile_student_iqa_14105_full_controlled_20260604
outputs/icaa_tf_native_savedmodel_20260516_125620
outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist05_rank0005_e3_20260517
outputs/icaa_export_safe_torchscript_20260514_153413
outputs/acut_product_vila_local_score_reasoning_v1
outputs/mdiqa_distillation_50k_smoke_targets_20260608
outputs/icaa_tf_native_tflite_fp32_20260516_115414
outputs/test_vila_aesthetic_direct_compare_20260504_153546
outputs/flutter_aesthetic_metadata_20260523
outputs/icaa_tf_native_savedmodel_20260516_130252
outputs/test_vila_nvila_eval_20260415_214918
outputs/aesthetic_full_paper_benchmark_gap_analysis_20260603_audit
outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517
outputs/rgnet_tflite_retry_20260507
outputs/eval_techiqa_guard_v1_stage1_frozen_20260520
outputs/mobile_alamp_v2_pretrained_4096_tflite
outputs/aesthetic_weight_lab_tflite_tool_20260508
outputs/alamp_paper_fusion_4096_chunked_graphgcn
outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522
outputs/alamp_paper_ava_classification_20260511
outputs/icaa_export_safe_original_parity_20260514_153033
outputs/acut_transparency_test_vila_20260416_gpu
outputs/smallrun_alamp_multipatch_teacher_repeatfix_20260524
outputs/hybrid_teacher_final_label_scope_decision_20260603
outputs/icaa_tf_native_tflite_fp32_20260516_125713
outputs/icaa_tf_native_savedmodel_20260516_123424
outputs/icaa_export_torchscript_20260514_144117
outputs/nima_preprocessing_fix_20260525
outputs/nima_quantized_vs_existing_comparison_20260525
outputs/mobile_student_iqa_14105_longer_controlled_20260604
outputs/test_vila_worker_verify_top10_20260416
outputs/dryrun_alamp_multipatch_teacher_default_compat_20260525
outputs/c6_topiq_preprocess_backend_ablation_20260529
outputs/acut_stage1_nima
outputs/icaa_tf_native_tflite_fp32_20260516_132355
outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608
outputs/alamp_paper_fusion_smoke_threshold_fix
outputs/rgnet_paper_reproduction_preflight_20260526
outputs/research_protocol_production_technical_eval_20260530
outputs/icaa_tf_native_savedmodel_20260516_130506
outputs/icaa_export_safe_full_eval_20260514_193638
outputs/mobile_student_iqa_v3_20260601
outputs/distill_full_aadb
outputs/icaa_export_safe_full_eval_20260514_190201
outputs/eval_topiq_rankcalib_v2_stage3_hparam_probe_20260517
outputs/icaa_official_eval_20260514_142417
outputs/c6_topiq_tensor_fingerprint_compare_20260529
outputs/aesthetic_paper_benchmark_old_vs_new_20260523
outputs/acut_transparency_test_vila_20260416
outputs/icaa_tf_native_tflite_fp32_20260516_131636
outputs/tflite
outputs/icaa_tf_native_savedmodel_20260516_123735
outputs/acut_full_stack
outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517
outputs/techiqa_guard_v1_direct_stage5_rank002_fpguard005_v2_20260520
outputs/icaa_tf_native_tflite_fp16_20260516_144023
outputs/icaa_export_torchscript_20260514_144141
outputs/smoke_musiq_spaq
outputs/icaa_tf_native_savedmodel_20260516_115119
outputs/icaa_tf_native_tflite_fp32_20260516_124642
outputs/paq2piq_flive_toolbox_split_verify_20260530
outputs/icaa_tf_native_tflite_fp32_20260516_124012
outputs/mdiqa_flive_image_test_sanity_20260601
outputs/icaa_tf_native_savedmodel_20260516_043428
outputs/acut_product_vila_local_format_check_v2
outputs/dryrun_alamp_multipatch_teacher_papercap4096_20260525
outputs/acut_model_training_code_audit_20260601
outputs/eval_distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523
outputs/icaa_tf_native_tflite_fp32_20260516_135745
outputs/acut_aesthetic_weight_multiobjective_20260603
outputs/acut_stage2_nima_musiq
outputs/mobile_alamp_v2_full_ava_smoke_20260519
outputs/icaa_tf_native_savedmodel_20260516_125735
outputs/acut_verify_now
outputs/hybrid_teacher_adjusted_calibration_design_20260603
outputs/audit_aesthetic_model_origin_20260504
outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524
outputs/probe_topiq_rankcalib_v2_stage1_regression_e15_20260517
outputs/aesthetic_teacher_reproduction_decision_20260526
outputs/topiq_teacher_smoke
outputs/smoke_topiq_rankcalib_v2_stage3_probe_mos1_dist03_rank0005_e3_20260517
outputs/icaa_tf_native_tflite_fp32_20260516_130213
outputs/probe_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517
outputs/paired_production_mdiqa_significance_expanded_20260603
outputs/icaa_export_safe_v2_tflite_debug_20260514_222350
outputs/icaa_tf_native_tflite_fp32_20260516_132950
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_smoke
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_smoke
outputs/icaa_export_safe_original_parity_20260514_194000
outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603
outputs/aesthetic_ensemble_weight_search_complete_20260603
outputs/student_distillation_manifest_14105_20260603
outputs/icaa_export_safe_torchscript_20260514_185840
outputs/mobile_student_iqa_v2_20260601
outputs/mobile_student_iqa_v2_training_loop_audit_20260601
outputs/icaa_tf_native_savedmodel_20260516_124346
outputs/gemma4_e4b_aesthetic3_smoke
outputs/icaa_tf_native_stage_f_20260516_112834
outputs/acut_product_nvila8b_probe_v4
outputs/icaa_tf_native_tflite_fp32_20260516_132602
outputs/predictions
outputs/icaa_export_safe_v2_tflite_fp32_parity_20260514_221722
outputs/research_protocol_baseline_eval_audit_20260530
outputs/full_topiq_lite_spaq_to_koniq_frozen_e30_20260517
outputs/rgnet_paper_consolidated_20260511
outputs/topiq_teacher_label_dryrun_20260601
```

## Existing report, audit, inventory, and claims files

Selection rule: files under `outputs/`, `reports/`, or `docs/` whose names are report-, audit-, inventory-, or claims-oriented. Model-specific logs and predictions are covered by the artifact manifest.

```text
outputs/mdiqa_flive_protocol_audit_20260601/paper_flive_claims.csv
outputs/mdiqa_flive_protocol_audit_20260601/report.md
outputs/rgnet_float32_retry_20260508/rgnet_float32_report.md
outputs/aesthetic_ensemble_production_precheck_20260603/production_precheck_report.md
outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/report.md
outputs/alamp_object_graph_subset_20260511/object_graph_subset_report.md
outputs/nima_fixed_preproc_eval_20260525/report.md
outputs/hybrid_teacher_adjusted_calibration_2k_20260603/report.md
outputs/alamp_paper_faithfulness_audit_20260511/alamp_paper_faithfulness_audit_report.md
outputs/acut_icaa_skip_weight_search_20260601/proposed_020_030_050_report.md
outputs/acut_icaa_skip_weight_search_20260601/report.md
outputs/eval_topiq_lite_ranking_probe_lam02_gap10_e10_20260517/report.md
outputs/eval_topiq_lite_ranking_probe_lam02_gap10_e10_20260517/final_report.md
outputs/hybrid_teacher_unknown_fallback_audit_20260603/data_inventory_notes.md
outputs/hybrid_teacher_unknown_fallback_audit_20260603/report.md
outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/report.md
outputs/topiq_teacher_label_expansion_50k_20260601/expansion_manifest_report.md
outputs/topiq_teacher_label_expansion_50k_20260601/inference_report.md
outputs/topiq_teacher_label_expansion_50k_20260601/report.md
outputs/topiq_flive_heuristic_gate_full_simulation_20260608/report.md
outputs/alamp_paper_fusion_1024_graphlite_20260513/fusion_smoke_report.md
outputs/rgnet_old_vs_paper_fp16_compare_20260514/report.md
outputs/acut_product_gold_weight_eval_20260603/report.md
outputs/technical_iqa_signal_comparison_20260604/report.md
outputs/icaa_int8_qat_audit_20260601/report.md
outputs/final_topiq_lite_mixed112_export_20260517/report.md
outputs/acut_weekly_research_journal_20260526/report_generation_log.md
outputs/full_iqa_iaa_model_design_audit_20260611/full_model_design_audit.md
outputs/rgnet_paper_aadb_tflite_export_20260514/parity/parity_report.md
outputs/eval_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517/report.md
outputs/eval_topiq_lite_ranking_probe_lam05_e12_20260517/report.md
outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525/report.md
outputs/gemini_api_teacher_aesthetic3_20260427_revised/visual_grounding_diagnostic/DIAGNOSTIC_REPORT.md
outputs/distortionguard_stageB3_flive_weighted_diagnosis_20260524/report.md
outputs/research_protocol_bounded_eval_20260530/report.md
outputs/icaa_performance_reverification_20260525/report.md
outputs/topiq_teacher_guard_design_20260531/report.md
outputs/rgnet_v1_aadb_pil_resize_retrain_20260524_tflite/parity_report.md
outputs/alamp_paper_mpnet_design_audit_20260511/alamp_paper_mpnet_design_audit_report.md
outputs/alamp_paper_mpnet_design_audit_20260511/patch_visualization_report.md
outputs/alamp_paper_mpnet_design_audit_20260511/patch_selection_report.md
outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/report.md
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/report.md
outputs/acut_aesthetic_benchmark_csv_audit_20260603/report.md
outputs/acut_aesthetic_benchmark_csv_audit_20260603/leakage_audit.md
outputs/paired_production_mdiqa_gpu_bottleneck_audit_20260603/report.md
outputs/acut_capstone_final_report_20260526/report.md
outputs/acut_capstone_final_report_20260526/report_generation_log.md
outputs/aesthetic_ensemble_weight_search_20260603/report.md
outputs/mobile_alamp_v2_pretrained_4096/report.md
outputs/mobile_alamp_v2_pretrained_4096/best_model_eval_report.md
outputs/topiq_mixed112_artifact_recovery_20260527/report.md
outputs/aesthetic_cross_predictions_for_weight_search_20260603/report.md
outputs/mobile_student_iqa_v4b_checkpoint_reeval_20260601/report.md
outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/gpu_preflight_report.md
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517/eval_report.md
outputs/distortionguard_hard_fp_v2_failure_taxonomy_20260524/report.md
outputs/topiq_flive_safety_gate_simulation_20260608/report.md
outputs/alamp_paper_fusion_smoke_20260513_fix/fusion_smoke_report.md
outputs/mobile_student_iqa_14105_smoke_20260603/report.md
outputs/mobile_alamp_v2_4096_smoke/report.md
outputs/final_topiq_lite_ranking_lam01_export_20260517/report.md
outputs/student_flive_hybrid_benchmark_20260604/report.md
outputs/mdiqa_variant_sanity_20260601/report.md
outputs/mobile_student_iqa_v2_error_analysis_20260601/report.md
outputs/eval_distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524/report.md
outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/report.md
outputs/official_protocol_benchmark_manifest_20260604/report.md
outputs/hybrid_teacher_label_dryrun_10k_20260603/report.md
outputs/alamp_gpu_acceleration_audit_20260603/report.md
outputs/icaa_ptq_candidates_20260601/review_20260601/dynamic_ptq_extended_report.md
outputs/icaa_ptq_candidates_20260601/review_20260601/dynamic_ptq_latency_report.md
outputs/icaa_ptq_candidates_20260601/review_20260601/report.md
outputs/icaa_ptq_candidates_20260601/model_inventory.csv
outputs/icaa_ptq_candidates_20260601/smoke_report.md
outputs/icaa_ptq_candidates_20260601/conversion_report.md
outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525/report.md
outputs/mobile_student_iqa_50k_smoke_20260608/report.md
outputs/topiq_flive_safety_signal_ablation_20260608/report.md
outputs/paper_protocol_technical_iqa_audit_20260530/report.md
outputs/mdiqa_official_protocol_expansion_audit_20260603/report.md
outputs/eval_alamp_multipatch_teacher_classweighted_block5_full_20260524/report.md
outputs/musiq_kaggle_smoke_20260601/report.md
outputs/alamp_paper_fusion_4096_chunked_graphlite/fusion_smoke_report.md
outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md
outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/report.md
outputs/technical_iqa_combo_benchmark_20260524/report.md
outputs/mobile_student_iqa_14105_tflite_export_20260604/report.md
outputs/alamp_rgnet_aadb_ava_comparison_20260513/model_family_comparison_report.md
outputs/alamp_paper_ava_v1_design_audit_20260511/alamp_v1_design_audit_report.md
outputs/research_protocol_image_test_eval_20260530/report.md
outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md
outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md
outputs/acut_model_core_training_code_excerpt_20260601/report.md
outputs/acut_model_core_training_code_excerpt_20260601/report_generation_log.md
outputs/student_training_manifest_design_20260601/report.md
outputs/student_training_manifest_design_20260601/student_training_manifest_report.md
outputs/eval_topiq_rankcalib_v2_stage1_regression_e15_20260517/report.md
outputs/rgnet_model_lineage_audit_20260513/rgnet_model_lineage_report.md
outputs/eval_full_topiq_lite_spaq_replacement_20260517/report.md
outputs/alamp_patch_fidelity_audit_20260525/report.md
outputs/hybrid_teacher_range_calibration_2k_20260603/report.md
outputs/color_aesthetics_audit_20260509/color_aesthetics_audit_report.md
outputs/mdiqa_smoke_20260601/report.md
outputs/acut_final_ratio_core_eval_20260603/report.md
outputs/alamp_dual_branch_gcn_failure_analysis_20260524/report.md
outputs/mobile_student_iqa_v4b_20260601_sanity/report.md
outputs/official_protocol_mdiqa_signal_benchmark_20260604/report.md
outputs/known_dataset_full_label_dryrun_20260603/report.md
outputs/eval_topiq_lite_mixed112_unfreeze_replacement_20260517/report.md
outputs/c6_android_wsl_parity_exif_fixed_20260528/report.md
outputs/eval_topiq_lite_flive_calibration_replacement_20260517/report.md
outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/report.md
outputs/research_protocol_image_test_eval_analysis_20260530/report.md
outputs/eval_topiq_rankcalib_v2_stage3_probeC_partial_unfreeze_20260520/report.md
outputs/alamp_paper_fusion_smoke_20260513_finalguard/fusion_smoke_report.md
outputs/mobile_alamp_v2_pretrained_smoke/report.md
outputs/aesthetic_mobile_paper_benchmark_20260603/report.md
outputs/mobile_alamp_v2_pretrained_probe_1024/report.md
outputs/alamp_detector_selection_20260511/detector_selection_report.md
outputs/rgnet_paper_aadb_mobile_export_audit_20260513/rgnet_mobile_export_audit_report.md
outputs/rgnet_paper_aadb_mobile_export_audit_20260513/resume_state_report.md
outputs/nima_underperformance_diagnosis_20260525/report.md
outputs/research_protocol_smoke_scoring_20260530/report.md
outputs/ava_pretrain_aadb_finetune_20260507/finetune_report.md
outputs/alamp_external_patch_validation_20260524/report.md
outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/report.md
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/report.md
outputs/alamp_paper_mpnet_patch_selector_v2_20260511/patch_selector_v2_comparison_report.md
outputs/alamp_paper_mpnet_patch_selector_v2_20260511/patch_selection_v2_report.md
outputs/alamp_paper_ava_v2_graph_lite_20260511/alamp_paper_ava_v2_graph_lite_report.md
outputs/ava_retrain_completion_audit_20260507/completion_audit_report.md
outputs/mdiqa_cuda_diagnosis_20260601/report.md
outputs/mdiqa_checkpoint_source_audit_20260601/report.md
outputs/mdiqa_student_final_no_go_report_20260611/report.md
outputs/mobile_student_iqa_v3_error_analysis_20260601/report.md
outputs/mobile_student_iqa_v4b_20260601_smoke_fixed/report.md
outputs/musiq_kaggle_sanity_fallback_20260601/report.md
outputs/c6_topiq_tensor_source_dimension_match_20260529/report.md
outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/report.md
outputs/alamp_paper_fusion_smoke_20260513/fusion_smoke_report.md
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/eval_report.md
outputs/alamp_layout_aware_teacher_plan_20260524/report.md
outputs/mpnet_gpu_env_audit_20260512/gpu_env_audit_report.md
outputs/research_protocol_bounded_eval_analysis_20260530/report.md
outputs/alamp_paper_strict_patch_selector_20260512/patch_selector_strict_report.md
outputs/topiq_lite_replacement_export_feasibility_20260517/report.md
outputs/eval_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517/report.md
outputs/eval_topiq_unified_full_flive_replacement_20260517/report.md
outputs/alamp_paper_mpnet_patch_selector_v4_20260512/manual_gemini_overlay_eval_report.md
outputs/alamp_paper_mpnet_patch_selector_v4_20260512/patch_selector_v4_report.md
outputs/alamp_paper_mpnet_patch_selector_v4_20260512/patch_selector_v4_comparison_report.md
outputs/alamp_paper_mpnet_patch_selector_v4_20260512/patch_size_audit_report.md
outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/report.md
outputs/hybrid_teacher_label_design_review_2k_20260603/report.md
outputs/alamp_dual_branch_gcn_teacher_preflight_20260524/report.md
outputs/topiq_teacher_performance_audit_20260601/report.md
outputs/paper_level_stage1_benchmark_alignment_20260523/report.md
outputs/mdiqa_correct_protocol_full_benchmark_20260601/report.md
outputs/eval_alamp_multipatch_teacher_full_ava_20260524_smoke/report.md
outputs/technical_iqa_signal_comparison_with_c6_20260604/report.md
outputs/alamp_mpnet_vs_aadb_alamp_review_20260513/alamp_mpnet_vs_aadb_alamp_report.md
outputs/final_topiq_mixed112_replacement_decision_20260520/report.md
outputs/c6_android_wsl_parity_exif_fixed_20260529/report.md
outputs/alamp_external_patch_full_conversion_20260524/report.md
outputs/rgnet_paper_aadb_regression_20260509/rgnet_paper_aadb_report.md
outputs/hybrid_teacher_label_design_review_20260603/report.md
outputs/mdiqa_student_gap_analysis_20260608/report.md
outputs/capstone_technical_iqa_baseline_summary_20260530/report_section.md
outputs/rgnet_paper_faithfulness_next_step_20260525/report.md
outputs/topiq_teacher_smoke_20260531/report.md
outputs/topiq_teacher_setup_audit_20260531.md
outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523/report.md
outputs/mdiqa_student_pre_final_audit_20260609/report.md
outputs/icaa_qat_attempt_20260601/qat_feasibility_report.md
outputs/icaa_qat_attempt_20260601/report.md
outputs/thesis_material_audit_20260607/report.md
outputs/audit_model_paper_basis_20260506/audit_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/mpnet_selector_compare_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu/v4_1024_gpu/threshold_calibration_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw23_p2mid/v4_1024_gpu_tuned_cw23_p2mid/threshold_calibration_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw22_p2short/v4_1024_gpu_tuned_cw22_p2short/threshold_calibration_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_best_val_loss/v4_1024_gpu_tuned_best_val_loss/threshold_calibration_report.md
outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned/v4_1024_gpu_tuned/threshold_calibration_report.md
outputs/known_dataset_full_label_coverage_audit_20260603/report.md
outputs/eval_topiq_lite_spaq_to_koniq_replacement_20260517/report.md
outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md
outputs/alamp_paper_fusion_4096_final_report.md
outputs/alamp_external_repo_audit_20260511/alamp_external_repo_audit_report.md
outputs/student_training_manifest_50k_20260601/student_training_manifest_report.md
outputs/rgnet_paper_v1_ablation_20260509/ablation_report.md
outputs/tech_iqa_guard_v1_dataset_design_20260520/report.md
outputs/mobile_student_iqa_v3_20260601_smoke/report.md
outputs/alamp_multipatch_teacher_improvement_plan_20260524/report.md
outputs/c6_android_parity_final_audit_20260530/report.md
outputs/mobile_student_iqa_14105_smoke_gpu_20260603/report.md
outputs/mobile_student_iqa_v1_20260601_smoke/report.md
outputs/acut_product_gold_weight_eval_test_vila_20260603/report.md
outputs/alamp_paper_mpnet_patch_selector_v3_20260512/patch_selector_v3_report.md
outputs/alamp_paper_mpnet_patch_selector_v3_20260512/patch_selector_v3_comparison_report.md
outputs/research_protocol_production_technical_eval_analysis_20260530/report.md
outputs/mobile_student_iqa_v2_20260601_smoke/report.md
outputs/paper_faithful_audit_20260509/paper_faithful_audit_report.md
outputs/mobile_alamp_v2_full_ava_20260519/report.md
outputs/mobile_alamp_v2_full_ava_20260519/best_model_eval_report.md
outputs/alamp_patch_overlay_audit_20260513/report.md
outputs/final_c6_log_only_readiness_20260528/report.md
outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/report.md
outputs/alamp_paper_fusion_4096_chunked_graphlite_smoke/fusion_smoke_report.md
outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest_report.md
outputs/topiq_teacher_guard_holdout_20260601/report.md
outputs/techiqa_guard_v1_hard_fp_mining_20260522/report.md
outputs/topiq_lite_fidelity_audit.md
outputs/aesthetic_mobile_paper_benchmark_20260603_audit/benchmark_validity_audit.md
outputs/mdiqa_distillation_50k_manifest_audit_20260608/report.md
outputs/ava_retrain_rgnet_alamp_20260506/ava_retrain_report.md
outputs/thesis_material_final_correction_20260607/final_corrected_report.md
outputs/thesis_material_final_correction_20260607/final_thesis_claims.csv
outputs/paq2piq_official_split_search_20260530/report.md
outputs/alamp_papercap_recipe_audit_20260525/report.md
outputs/alamp_external_integration_feasibility_audit_20260524/report.md
outputs/mpnet_gpu_cudnn_audit_20260513/gpu_cudnn_audit_report.md
outputs/alamp_dual_branch_4096_comparison_20260524/report.md
outputs/alamp_dual_branch_4096_comparison_20260524/eval_multipatch_only_4096/report.md
outputs/official_protocol_technical_signal_benchmark_20260604/base_run/report.md
outputs/official_protocol_technical_signal_benchmark_20260604/report.md
outputs/official_protocol_technical_signal_benchmark_20260604/smoke_test/report.md
outputs/official_protocol_technical_signal_benchmark_20260604/c6_run/report.md
outputs/mdiqa_distillation_50k_full_targets_20260608/report.md
outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.md
outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.md
outputs/rgnet_paper_v1_ablation_full_candidates_20260510/full_candidate_report.md
outputs/mobile_student_iqa_v2_20260601_fixed/report.md
outputs/alamp_paper_fusion_1024_comparison_20260513/report.md
outputs/final_c6_topiq_koniq_technical_score_decision_20260527/report.md
outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/report.md
outputs/mobile_student_iqa_v1_20260601/report.md
outputs/final_techiqa_guard_v1_nogo_20260522/report.md
outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/report.md
outputs/paper_level_aesthetic_gap_audit_20260523/report.md
outputs/paired_production_mdiqa_significance_20260603/report.md
outputs/paired_production_mdiqa_significance_20260603/production_scoring_source_audit.md
outputs/aesthetic_weight_lab_structure_audit_20260508/structure_audit_report.md
outputs/koniq_android_wsl_parity_variant_diagnosis_20260528/report.md
outputs/mdiqa_sanity_20260601/report.md
outputs/rgnet_mobile_resize_finetune_preflight_20260524/report.md
outputs/mdiqa_wsl_feasibility_audit_20260601/report.md
outputs/nima_icaa_paper_level_eval_20260525/report.md
outputs/mobile_alamp_v2_4096_gpu_smoke/report.md
outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526/report.md
outputs/rgnet_faithful_architecture_audit_20260524/report.md
outputs/alamp_paper_fusion_1024_graphgcn_20260513/fusion_smoke_report.md
outputs/final_research_protocol_technical_iqa_baseline_report_20260530/report.md
outputs/mobile_student_iqa_v2_training_loop_verify_20260601/report.md
outputs/research_protocol_iqa_manifests_20260530/report.md
outputs/acut_weekly_research_journal_v2_20260526/report_generation_log.md
outputs/musiq_kaggle_sanity_20260601/report.md
outputs/acut_model_core_code_audit_20260601/report.md
outputs/acut_model_core_code_audit_20260601/report_generation_log.md
outputs/technical_teacher_hybrid_design_audit_20260603/report.md
outputs/alamp_paper_mpnet_4096_expansion/v4_4096/report.md
outputs/alamp_paper_mpnet_4096_expansion/v4_4096/eval/v4_4096/threshold_calibration_report.md
outputs/rgnet_paper_ava_classification_20260510/threshold_calibration/threshold_calibration_report.md
outputs/rgnet_paper_ava_classification_20260510/rgnet_paper_ava_report.md
outputs/eval_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517/report.md
outputs/mobile_alamp_v2_1024_probe/report.md
outputs/paq2piq_external_official_split_search_20260530/report.md
outputs/mobile_student_iqa_v4b_20260601/report.md
outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517/full_train/lse_r4/eval_report.md
outputs/musiq_kaggle_failure_diagnosis_20260601/report.md
outputs/rgnet_paper_v1_aadb_regression_20260509/rgnet_paper_v1_aadb_report.md
outputs/mobile_student_iqa_v4a_20260601/report.md
outputs/eval_topiq_lite_mixed_replay_replacement_20260517/report.md
outputs/eval_distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524/report.md
outputs/mdiqa_official_variant_sanity_20260601/report.md
outputs/musiq_wsl_kaggle_audit_20260601/report.md
outputs/c6_android_wsl_parity_20260527/report.md
outputs/mobile_student_iqa_14105_short_train_20260604/report.md
outputs/eval_alamp_multipatch_teacher_full_ava_20260524/report.md
outputs/distortionguard_iqa_v1_design_20260523/report.md
outputs/topiq_teacher_validation_20260531/report.md
outputs/topiq_teacher_validation_20260531/validation_manifest_report.md
outputs/alamp_salient_detector_audit_20260512/salient_detector_audit_report.md
outputs/paper_level_teacher_reproduction_plan_20260524/report.md
outputs/thesis_material_correction_audit_20260607/corrected_thesis_claims.csv
outputs/thesis_material_correction_audit_20260607/corrected_report.md
outputs/aesthetic_ensemble_weight_search_complete_20260603_audit/weight_search_audit.md
outputs/mobile_student_iqa_14105_full_controlled_20260604/report.md
outputs/mdiqa_distillation_50k_smoke_targets_20260608/report.md
outputs/rgnet_tflite_retry_20260507/rgnet_tflite_retry_report.md
outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/report.md
outputs/mobile_alamp_v2_pretrained_4096_tflite/parity_report.md
outputs/aesthetic_weight_lab_tflite_tool_20260508/handoff_report.md
outputs/alamp_paper_fusion_4096_chunked_graphgcn/fusion_smoke_report.md
outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522/report.md
outputs/alamp_paper_ava_classification_20260511/alamp_paper_ava_report.md
outputs/hybrid_teacher_final_label_scope_decision_20260603/report.md
outputs/nima_preprocessing_fix_20260525/report.md
outputs/nima_quantized_vs_existing_comparison_20260525/report.md
outputs/mobile_student_iqa_14105_longer_controlled_20260604/report.md
outputs/c6_topiq_preprocess_backend_ablation_20260529/report.md
outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md
outputs/alamp_paper_fusion_smoke_threshold_fix/fusion_smoke_report.md
outputs/rgnet_paper_reproduction_preflight_20260526/report.md
outputs/research_protocol_production_technical_eval_20260530/production_formula_audit.md
outputs/research_protocol_production_technical_eval_20260530/report.md
outputs/mobile_student_iqa_v3_20260601/report.md
outputs/eval_topiq_rankcalib_v2_stage3_hparam_probe_20260517/report.md
outputs/c6_topiq_tensor_fingerprint_compare_20260529/report.md
outputs/aesthetic_paper_benchmark_old_vs_new_20260523/report.md
outputs/aesthetic_paper_benchmark_old_vs_new_20260523/verification_report.md
outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517/report.md
outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517/final_report.md
outputs/paq2piq_flive_toolbox_split_verify_20260530/verification_report.md
outputs/mdiqa_flive_image_test_sanity_20260601/report.md
outputs/acut_model_training_code_audit_20260601/report.md
outputs/acut_model_training_code_audit_20260601/report_generation_log.md
outputs/eval_distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523/report.md
outputs/acut_aesthetic_weight_multiobjective_20260603/report.md
outputs/mobile_alamp_v2_full_ava_smoke_20260519/report.md
outputs/audit_aesthetic_model_origin_20260504/audit_report.md
outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/report.md
outputs/aesthetic_teacher_reproduction_decision_20260526/report.md
outputs/paired_production_mdiqa_significance_expanded_20260603/production_baseline_discrepancy_audit.md
outputs/paired_production_mdiqa_significance_expanded_20260603/report.md
outputs/paired_production_mdiqa_significance_expanded_20260603/tf_gpu_environment_audit.md
outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/tflite_constant_output_debug_report.md
outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/alamp_adaptive_patch_benchmark_report.md
outputs/aesthetic_ensemble_weight_search_complete_20260603/report.md
outputs/student_distillation_manifest_14105_20260603/report.md
outputs/mobile_student_iqa_v2_20260601/report.md
outputs/mobile_student_iqa_v2_training_loop_audit_20260601/report.md
outputs/research_protocol_baseline_eval_audit_20260530/report.md
outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest_report.md
outputs/topiq_teacher_label_dryrun_20260601/inference_report.md
outputs/topiq_teacher_label_dryrun_20260601/report.md
reports/advisor_weekly_report/tables/model_inventory.csv
reports/advisor_weekly_report/final_professor_report.md
reports/advisor_weekly_report/full_audit_index.md
reports/advisor_report/audit_summary.md
reports/advisor_report/advisor_report.md
docs/github_upload_audit_20260605.md
docs/repository_inventory_20260605.md
docs/icaa_official_repo_dataset_audit.md
docs/full_repository_directory_audit_20260605.md
docs/color_aesthetics_kickoff_audit.md
```

## Snapshot limitations

- Untracked-file status is complete as reported by Git, but ignored content such as most environment internals is not enumerated by Git.
- The directory tree is structural and depth-limited. It does not imply artifact lineage or model role.
- Existing output names are evidence of prior work only. Classification and association require corroborating code, metadata, reports, or runtime references.

