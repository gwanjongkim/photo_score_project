# Source disagreements and ambiguities

| Topic | Interpretation A | Interpretation B | Authority and treatment |
|---|---|---|---|
| AVA split | NIMA says random 80/20 | RGNet, A-LAMP and AADB paper use the established ~230k/~20k literature split | Both are primary and method-specific. Never call them one canonical split without filename proof. |
| RGNet on AADB | RGNet paper uses scalar MSE and SRCC | AADB’s originating model uses Euclidean + pairwise ranking + attributes/content | RGNet paper is authoritative for RGNet. Attribute/ranking losses are essential to the AADB ranking method, not RGNet. |
| A-LAMP statistics | Sec. 3.1.2/equation says `{max, mean}` | Footnote says extensive experiments found `max, min` best | Internal paper inconsistency. Record chosen interpretation; official source code/supplement was inaccessible. Confidence LOW for exact pair. |
| A-LAMP “attention” | User/project terminology may call fusion attention | Paper uses adaptive patch selection and layout attribute graphs; no generic neural attention module is the defining component | Paper is authoritative. Describe the actual mechanisms rather than retrofitting transformer attention. |
| ICAA heads | Table 4 distinguishes color-only `Ours*1` and joint holistic `Ours*2` | Inventory/local naming may imply several attribute heads or label dimensions | Paper/repo are authoritative; no fixed 30-head interpretation is supported. Exact return arity remains unresolved. |
| MUSIQ KonIQ split | MUSIQ randomly samples 80/20 for each run | KonCept512 uses the KonIQ authors’ published split | Method-specific protocols. A score must cite which protocol and number of runs. |
| MUSIQ input | Full MUSIQ is native + two ARP scales | MUSIQ-single is native only | Both paper variants are valid, but only the former may be called the default full model. |
| FLIVE versus PaQ-2-PiQ | FLIVE is a dataset with full and patch MOS | PaQ-2-PiQ is the region model introduced with it | Dataset-trained mobile regressors must not inherit the architecture name automatically. |
| MDIQA paper/code | Paper is arXiv v1 and describes the normative concept | Official README supplies staged scripts/options and an `open_clip` patch | Treat code details as official implementation requirements; record any future paper-version change rather than silently merging it. |
| Stage 5 | Project label denotes a student/teacher workflow | Knowledge distillation literature is a broad family | No identity equivalence. Stage 5 has no normative paper baseline until a specific cited method is supplied. |
| LCC versus PLCC | ICAA paper labels `LCC` | Other IQA papers use `PLCC` | Do not normalize labels silently; report the paper’s label and formula/code if comparison needs equivalence. |
