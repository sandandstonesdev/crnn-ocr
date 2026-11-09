## Co to jest

CRNN (CNN + BiLSTM + CTC) do rozpoznawania polskich linii tekstu, trenowany na [ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines).

Architektura: **loader + CRNN + CTC loss + dekoder + metryki**

## Struktura katalogów

```
crnn-ocr/
│
├── README.md                  # opis, setup, jak odpalić trening
├── environment.yml            # specyfikacja środowiska conda
├── .gitignore
│
├── configs/
│   └── default.yaml           # wszystkie hiperparametry w jednym miejscu
│
├── data/
│   ├── __init__.py
│   ├── charset.py             # budowa alfabetu z datasetu (blank=0)
│   ├── dataset.py             # LineDataset + collate (padding, lens)
│   └── transforms.py          # resize do H=32, augmentacje (opcjonalnie)
│
├── model/
│   ├── __init__.py
│   └── crnn.py                # klasa CRNN (CNN → sekwencja → BiLSTM → FC)
│
├── decoding/
│   ├── __init__.py
│   └── greedy.py              # greedy CTC decode (potem: beam.py)
│
├── metrics.py                 # CER, WER, exact match (editdistance)
│
├── train.py                   # pętla treningu + CTC loss + TensorBoard
├── evaluate.py                # ewaluacja na val/test → metryki + przykłady
├── infer.py                   # predykcja na pojedynczym obrazie (pipeline)
│
├── runs/                      # logi TensorBoard
├── checkpoints/               # zapisywane modele
└── tests/                     # testy jednostkowe (opcjonalnie)
```

`runs/` i `checkpoints/` są ignorowane przez git (artefakty treningu, nie kod).

## Setup

```bash
conda env create -f environment.yml
conda activate crnn-ocr
```

## Trening

```bash
python train.py --config configs/default.yaml
```

Logi TensorBoard: `tensorboard --logdir runs/`

## Ewaluacja

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

## Wyniki

*(do uzupełnienia po pierwszych treningach: CER / WER / exact match na zbiorze walidacyjnym i testowym)*