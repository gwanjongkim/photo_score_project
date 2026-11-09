# Paper-reported results reference

These are literature values, not local results. A value is comparable only when dataset, split, target, architecture variant and aggregation match.

| Method | Dataset/task | Reported result | Exact location | Status/confidence |
|---|---|---|---|---|
| NIMA MobileNet | AVA | accuracy .8036; LCC(mean) .518; SRCC(mean) .510; EMD .081 | S01 Table I | VERIFIED_PRIMARY_SOURCE / HIGH |
| NIMA VGG16 | AVA | accuracy .8060; LCC .610; SRCC .592; EMD .052 | S01 Table I | VERIFIED_PRIMARY_SOURCE / HIGH |
| NIMA Inception-v2 | AVA | accuracy .8151; LCC .636; SRCC .612; EMD .050 | S01 Table I | VERIFIED_PRIMARY_SOURCE / HIGH |
| AADB Reg+Rank+Att+Cont | AVA | SRCC .5581; accuracy 77.33% | S03 Table 3 | VERIFIED_PRIMARY_SOURCE / HIGH |
| AADB best model | AADB | SRCC .6782 | S03 Table 2 and Table 6 | VERIFIED_PRIMARY_SOURCE / HIGH |
| AADB cross-dataset | train AADB→test AVA; train AVA→test AADB | .1566; .3191 SRCC | S03 Table 7 | VERIFIED_PRIMARY_SOURCE / HIGH |
| RGNet | AADB | SRCC .7104 | S04 Table 5 | VERIFIED_PRIMARY_SOURCE / HIGH |
| RGNet | AVA classification | 84.61% accuracy (paper comparison table) | S04 Table 4 | VERIFIED_PRIMARY_SOURCE / MEDIUM |
| A-LAMP | AVA binary | 82.5% accuracy; .92 F-measure | S05 Table 2 | VERIFIED_PRIMARY_SOURCE / HIGH |
| New-MP-Net | AVA binary | 81.7% accuracy; .91 F-measure | S05 Tables 1–2 | VERIFIED_PRIMARY_SOURCE / HIGH |
| ICAA DAT (`Ours*1`) | SPAQ colorfulness benchmark | SRCC .770; LCC .791 | S06 Table 4 | VERIFIED_PRIMARY_SOURCE / HIGH |
| MUSIQ | PaQ-2-PiQ full-size test | SRCC .646; PLCC .739 | S08 Table 1 | VERIFIED_PRIMARY_SOURCE / HIGH |
| MUSIQ | KonIQ-10k | SRCC .916; PLCC .928 | S08 Table 2, 10-run mean | VERIFIED_PRIMARY_SOURCE / HIGH |
| MUSIQ | SPAQ | SRCC .917; PLCC .921 | S08 Table 3 | VERIFIED_PRIMARY_SOURCE / HIGH |
| MUSIQ | AVA | accuracy .815; MSE .242; SRCC .726; PLCC .738 | S08 Table 4 | VERIFIED_PRIMARY_SOURCE / HIGH |
| KonCept512 | KonIQ-10k | SROCC .921 (paper headline/test result) | S10 abstract/result table | VERIFIED_PRIMARY_SOURCE / MEDIUM |
| KonCept512 cross-dataset | LIVE-in-the-Wild | SROCC .825 | S10 abstract/cross-dataset table | VERIFIED_PRIMARY_SOURCE / MEDIUM |

MDIQA and TOPIQ numeric tables were not transcribed because the exact paper table cells were not captured with adequate confidence. They remain `UNVERIFIED`; no value from a README, local report, or third-party benchmark is substituted.
