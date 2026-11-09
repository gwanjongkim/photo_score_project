# Official repository requirements

Only repositories actually opened are marked verified. File paths describe upstream repositories, never the local project.

| Method | Official repository | Verified paths/symbols | Normative use | Status/confidence |
|---|---|---|---|---|
| NIMA | No author repository was linked by the opened paper | none | Paper, not third-party Keras ports, is normative. Backbone preprocessing remains version-sensitive. | UNVERIFIED / MEDIUM |
| RGNet | No author-maintained repository was located in opened primary sources | none | Paper equations and architecture are the only verified baseline. | UNVERIFIED / HIGH |
| A-LAMP | No author-maintained code was opened | none | Caffe implementation details not stated in paper remain unresolved. | UNVERIFIED / HIGH |
| AADB | Project page is named in the authors’ poster, but no maintained repository was opened | none | Do not elevate unofficial ports. | SOURCE_INACCESSIBLE / MEDIUM |
| ICAA/DAT | https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment | `ICAA17K_code/models/`, `dataset.py`, `main_nni.py`, `config.py`, `config.yml`, `option.py`, `search_space.json`; README requires AVA pretraining and `nnictl create --config config.yml -p 8999` | Official code may resolve paper omissions. It may not override the paper’s task definition without recording a disagreement. | VERIFIED_OFFICIAL_CODE / HIGH |
| MUSIQ | https://github.com/google-research/google-research/tree/master/musiq | `musiq/` package tree and README | Use for exact ARP resize, hashing, padding/masking, normalization and released checkpoint inference. | VERIFIED_OFFICIAL_CODE / HIGH |
| PaQ-2-PiQ | https://github.com/baidut/PaQ-2-PiQ | repository README, `paq2piq/` package, released weights/training and evaluation notebooks visible in tree | Official implementation defines ROI construction and full-/patch-set evaluation details omitted from prose. | VERIFIED_OFFICIAL_CODE / HIGH |
| MDIQA | https://github.com/YaoShunyu19/MDIQA | `MDIQA/train_shared.py`, `MDIQA/train_MDIQA.py`, `MDIQA/options_MDIQA/aesthetic/train_spaq_5_tech_CFANet.yml`, `.../train_para_4_aes_CFANet.yml`, `.../benchmark_experiments/train_MDIQA_koniq.yml`, `.../train_MDIQA_xd_koniq.yml` | Reproduction requires both staged training and the specified option file; README also patches `open_clip/modified_resnet.py`. | VERIFIED_OFFICIAL_CODE / HIGH |
| TOPIQ | https://github.com/chaofengc/IQA-PyTorch | `pyiqa/archs/topiq_arch.py`, `pyiqa/default_model_configs.py`/metric configs and weight registry | Distinguish NR/FR and dataset-specific released weights. Toolbox defaults are official-code requirements, not necessarily paper-essential. | VERIFIED_OFFICIAL_CODE / MEDIUM |
| KonCept512 | Official dataset page links a GitHub repository, but its file tree was not opened in this phase | none | Paper and official dataset page remain normative; code-only details unresolved. | SOURCE_INACCESSIBLE / MEDIUM |

Repository-version rule: record upstream URL, commit/tag, checkpoint URL/hash, dependency versions and any local patch. “Official repository” does not make later code changes retroactively part of the published paper; differences belong in `source_disagreements.md`.
