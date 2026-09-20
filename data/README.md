# Datasets

The three benchmarks used in this study are publicly available. This repository does not redistribute them. Follow the steps below to populate `data/raw/` before running the pipeline.

## BoT-IoT

- **Source:** https://research.unsw.edu.au/projects/bot-iot-dataset
- **Citation:** Koroniotis et al., *Future Generation Computer Systems* 100 (2019) 779–796
- **Subset used:** 5% extract (3,668,522 records, 5 classes)
- **Placement:** `data/raw/bot_iot/Bot-IoT-5percent.csv`

## N-BaIoT

- **Source:** https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
- **Citation:** Meidan et al., *IEEE Pervasive Computing* 17 (3) (2018) 12–22
- **Subset used:** All 11 categories aggregated from nine devices (7,062,606 records)
- **Placement:** `data/raw/n_baiot/` — one CSV per device, then aggregated by the pipeline

## IoTID20

- **Source:** https://sites.google.com/view/iotnetworktrafficclassificationincicidsiotid20/home
- **Citation:** Ullah and Mahmoud, *Canadian AI 2020*, LNCS 12109
- **Subset used:** Full dataset (625,783 records, 5 classes)
- **Placement:** `data/raw/iot_id20/IoTID20.csv`

## Split Definition

The stratified 70/15/15 train/validation/test split (random seed 42) is committed as JSON for reproducibility:

- `data/splits/bot_iot_split.json`
- `data/splits/n_baiot_split.json`
- `data/splits/iot_id20_split.json`

Each file records the row indices (in the preprocessed, pre-SMOTE dataframe) assigned to each partition.

## Preprocessing

Run:

```bash
python -m src.preprocessing --dataset all
```

This applies the eight-stage pipeline described in Section 3.3 of the paper:

1. Cleaning (dedup, drop high-cardinality columns)
2. Label encoding
3. 70/15/15 stratified split (seed 42)
4. SelectKBest with mutual information (k=32), fit on train only
5. StandardScaler, fit on train only
6. SMOTE oversampling on train only
7. Sliding-window construction (length 8, shape (N, 8, 32))
8. Uniform input representation

Outputs land in `data/processed/` and are gitignored.
