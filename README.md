# Wykorzystanie sieci syjamskich do identyfikacji indywidualnego psa na podstawie analizy obrazu

Praca licencjacka — Informatyka SWPS

🇬🇧 English version below

## Opis

Projekt porównuje dwa podejścia do rozpoznawania konkretnego psa na zdjęciach:
- **Klasyfikator binarny** — klasyfikacja pojedynczego zdjęcia (pies badany / inny pies)
- **Sieć syjamska** — porównanie par zdjęć i określenie ich podobieństwa

Oba modele wykorzystują architekturę ResNet-50 z wagami przetrenowanymi na ImageNet (transfer learning).

## Wyniki

Oba podejścia osiągnęły dokładność na poziomie ~99% na zbiorze testowym (216 zdjęć, 4 uruchomienia każdego modelu).

| Model | Accuracy | Liczba błędów |
|---|---|---|
| Klasyfikator binarny | 98.61–99.07% | 2–3 |
| Sieć syjamska | 97.22–99.54% | 1–6 |

## Struktura projektu"

```
├── binary_classification.py   # Podejście 1: klasyfikator binarny
├── siamese_network.py         # Podejście 2: sieć syjamska
├── requirements.txt           # Wymagane biblioteki
├── .gitignore
└── README_.md              
```

## Zbiór danych

Zbiór danych składa się z:
- **434 zdjęcia** psa badanego (3 kategorie: standard, young, difficult)
- **1000 zdjęć** 15 ras ze Stanford Dogs Dataset (3 kategorie: similar, normal, different)

Dataset dostępny na Google Drive: https://drive.google.com/drive/folders/1Bo02Lk2ujt3u2QScbOK-Q_JJxmpqDtfQ?usp=share_link

## Sprzęt

- Python 3.13
- MacBook Pro, Apple M4 Pro, 24 GB RAM

## Instalacja i uruchomienie

```bash
pip install -r requirements.txt
python binary_classification.py
python siamese_network.py
```

## Autor

Julia Nowak — praca licencjacka, 2026

## 🇬🇧 English version

# Siamese Networks for Individual Dog Identification Based on Image Analysis

Bachelor's thesis — Computer Science, SWPS

## Description

This project compares two approaches to recognizing a specific dog in photos:
- **Binary classifier** — classifies a single image (target dog / other dog)
- **Siamese network** — compares pairs of images and determines their similarity

Both models use ResNet-50 architecture with ImageNet pretrained weights (transfer learning).

## Results

Both approaches achieved ~99% accuracy on the test set (216 images, 4 runs per model).

| Model | Accuracy | Errors |
|---|---|---|
| Binary classifier | 98.61–99.07% | 2–3 |
| Siamese network | 97.22–99.54% | 1–6 |

## Project structure
```
├── binary_classification.py   # Approach 1: binary classifier
├── siamese_network.py         # Approach 2: siamese network
├── requirements.txt           # Required libraries
├── .gitignore
└── README_.md              
```

## Dataset

The dataset consists of:
- **434 images** of the target dog (3 categories: standard, young, difficult)
- **1000 images** of 15 breeds from Stanford Dogs Dataset (3 categories: similar, normal, different)

Dataset available on Google Drive: https://drive.google.com/drive/folders/1Bo02Lk2ujt3u2QScbOK-Q_JJxmpqDtfQ?usp=share_link

## Requirements

- Python 3.13
- MacBook Pro, Apple M4 Pro, 24 GB RAM

## Installation and usage

```bash
pip install -r requirements.txt
python binary_classification.py
python siamese_network.py
```

## Author

Julia Nowak — Bachelor's thesis, 2026


