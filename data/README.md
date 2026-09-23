# Datasets

The datasets are not redistributed with this repository. Download them from Kaggle into this `data/`
folder; every script and notebook reads from here by default.

The quickest route is the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (requires a Kaggle API token):

```bash
pip install kaggle
kaggle datasets download apollo2506/eurosat-dataset          -p data --unzip
kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data --unzip
kaggle datasets download emmarex/plantdisease                -p data --unzip
```

Downloading through the Kaggle website and extracting into `data/` produces the same layout.

| Case study | Kaggle dataset | Original source | License |
| --- | --- | --- | --- |
| EuroSAT | [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) | Helber et al., 2019 | MIT |
| Pneumonia | [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Kermany et al., 2018 | CC BY 4.0 |
| Leaf disease | [`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease) | Hughes & Salathé, 2015 (PlantVillage) | See the original PlantVillage release |

## Expected layout

```
data/
├── EuroSATallBands/
│   ├── label_map.json
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── AnnualCrop/AnnualCrop_1.tif
│   └── ... (10 class folders of 13-band GeoTIFFs)
├── chest_xray/
│   ├── train/{NORMAL,PNEUMONIA}/*.jpeg
│   ├── val/{NORMAL,PNEUMONIA}/*.jpeg
│   └── test/{NORMAL,PNEUMONIA}/*.jpeg
└── PlantVillage/
    ├── Pepper__bell___Bacterial_spot/*.JPG
    └── ... (15 class folders)
```

Only `EuroSATallBands/` is used from the EuroSAT archive (the RGB `EuroSAT/` folder can be deleted).
The chest X-ray archive also contains a duplicate `chest_xray/chest_xray/` folder and `__MACOSX/`
metadata; neither is used.

## Generated folders

These are created automatically on first run:

- `data/tfrecords/eurosat/` — TFRecord caches of the normalized EuroSAT splits.
- `data/chest_xray_rebalanced/` — the chest X-ray images re-split into stratified train/val/test sets.
  The original `val/` split contains only 16 images, so all images are pooled and split 85/15 into
  train+val/test, then train+val is split 85/15 into train/val (`random_state=42`).

## Splits

- **EuroSAT** uses the provided `train.csv`, `validation.csv` and `test.csv`.
- **Pneumonia** uses the re-split described above.
- **Leaf disease** shuffles the full directory listing (`random_state=42`) and takes 80/10/10 for
  train/val/test.

## Citations

- P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land
  Use and Land Cover Classification.* IEEE JSTARS 12(7), 2217–2226, 2019. doi:10.1109/JSTARS.2019.2918242
- D. Kermany, K. Zhang, M. Goldbaum. *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for
  Classification.* Mendeley Data, V2, 2018. doi:10.17632/rscbjbr9sj.2
- D. P. Hughes, M. Salathé. *An open access repository of images on plant health to enable the development of
  mobile disease diagnostics.* arXiv:1511.08060, 2015.
