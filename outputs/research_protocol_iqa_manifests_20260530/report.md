# Research-Protocol IQA Manifests for A-cut

## 1. Summary
Generated four benchmark artifacts:
- `benchmark_koniq10k_musiq_80_20_10runs.csv`
- `benchmark_spaq_official_split.csv`
- `benchmark_flive_paq2piq_toolbox_split.csv`
- `benchmark_all_research_protocol_iqa.csv`

The builder preserved the confirmed local protocols and kept FLIVE/PaQ-2-PiQ labeled as IQA-Toolbox/PyIQA standardized metadata, not original-author official split.

## 2. Confirmed Protocols Used
- KonIQ-10k MUSIQ-style 80/20 repeated 10 runs
- SPAQ official split
- FLIVE/PaQ-2-PiQ IQA-Toolbox/PyIQA standardized official_split

## 3. KonIQ-10k Manifest
Source annotation file: `data/raw/koniq10k/koniq10k_scores_and_distributions.csv`
MOS scale: raw MOS from the annotation file, normalized to 0..100 by multiplying by 20.0.
Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

Row counts per run.
```
run_id split  count
run_01  test   2015
run_01 train   8058
run_02  test   2015
run_02 train   8058
run_03  test   2015
run_03 train   8058
run_04  test   2015
run_04 train   8058
run_05  test   2015
run_05 train   8058
run_06  test   2015
run_06 train   8058
run_07  test   2015
run_07 train   8058
run_08  test   2015
run_08 train   8058
run_09  test   2015
run_09 train   8058
run_10  test   2015
run_10 train   8058
```

## 4. SPAQ Manifest
Source annotation files: `external/icaa_official_repo/SPAQ_code/dataset/SPAQ/spaqTrain.csv` and `external/icaa_official_repo/SPAQ_code/dataset/SPAQ/spaqTest.csv`
MOS scale: raw MOS from the official split CSVs, normalized to 0..100 by multiplying by 10.0.

Split counts.
```
split  count
 test   1125
train  10000
```

## 5. FLIVE / PaQ-2-PiQ Manifest
Source metadata basis: `outputs/paq2piq_flive_toolbox_split_verify_20260530/meta_info_FLIVEDataset.csv` via `outputs/paq2piq_flive_toolbox_split_verify_20260530/benchmark_flive_paq2piq_toolbox_split_preview.csv`
Protocol label: `IQA-Toolbox/PyIQA standardized FLIVE/PaQ-2-PiQ official_split`
Score basis: verified local label match from the prior toolbox split verification, with raw MOS already on a 0..100 scale.
Warning: this is toolbox-standardized metadata and is not directly proven to be the original-author official split.

Total rows: 159240

Image and patch split counts.
```
content_type split  count
       image  test   1827
       image train  30256
       image   val   7727
       patch  test   5481
       patch train  90768
       patch   val  23181
```

## 6. Combined Manifest
`benchmark_all_research_protocol_iqa.csv` concatenates the three dataset manifests into one file using the shared schema:
`dataset, protocol, run_id, split, content_type, image_id, image_path, quality_score_raw, quality_score_0_100, source_annotation_file, notes`.

## 7. Validation Checks
- row counts by dataset/protocol/split/run_id:
```
 dataset                                                      protocol split   run_id  count
   flive IQA-Toolbox/PyIQA standardized FLIVE/PaQ-2-PiQ official_split  test official   7308
   flive IQA-Toolbox/PyIQA standardized FLIVE/PaQ-2-PiQ official_split train official 121024
   flive IQA-Toolbox/PyIQA standardized FLIVE/PaQ-2-PiQ official_split   val official  30908
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_01   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_01   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_02   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_02   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_03   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_03   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_04   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_04   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_05   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_05   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_06   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_06   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_07   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_07   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_08   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_08   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_09   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_09   8058
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs  test   run_10   2015
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs train   run_10   8058
    spaq                                           SPAQ official split  test official   1125
    spaq                                           SPAQ official split train official  10000
```
- missing image count: 0
- missing score count: 0
- duplicate image_id count per dataset/run/split:
```
 dataset   run_id split  duplicate_image_id_count
   flive official  test                         0
   flive official train                         0
   flive official   val                         0
koniq10k   run_01  test                         0
koniq10k   run_01 train                         0
koniq10k   run_02  test                         0
koniq10k   run_02 train                         0
koniq10k   run_03  test                         0
koniq10k   run_03 train                         0
koniq10k   run_04  test                         0
koniq10k   run_04 train                         0
koniq10k   run_05  test                         0
koniq10k   run_05 train                         0
koniq10k   run_06  test                         0
koniq10k   run_06 train                         0
koniq10k   run_07  test                         0
koniq10k   run_07 train                         0
koniq10k   run_08  test                         0
koniq10k   run_08 train                         0
koniq10k   run_09  test                         0
koniq10k   run_09 train                         0
koniq10k   run_10  test                         0
koniq10k   run_10 train                         0
    spaq official  test                         0
    spaq official train                         0
```
- score range after normalization:
```
 dataset                                                      protocol quality_score_raw        quality_score_0_100      
                                                                                     min    max                 min   max
   flive IQA-Toolbox/PyIQA standardized FLIVE/PaQ-2-PiQ official_split          0.000000 100.00            0.000000 100.0
koniq10k                  KonIQ-10k MUSIQ-style 80/20 repeated 10 runs          1.096154   4.31           21.923077  86.2
    spaq                                           SPAQ official split          0.200000   9.60            2.000000  96.0
```
- FLIVE image and patch split counts:
```
content_type split  count
       image  test   1827
       image train  30256
       image   val   7727
       patch  test   5481
       patch train  90768
       patch   val  23181
```
- datasets included: ['flive', 'koniq10k', 'spaq']
- no AADB/AVA/ICAA17K included: yes

## 8. Next Action
A. Evaluate existing production technical scoring baselines on these manifests

## 9. Forbidden Actions
- Do not call the toolbox split original-author official.
- Do not use AADB/AVA/ICAA17K as technical IQA labels.
- Do not replace production technical_score.
- Do not train yet unless explicitly requested.
