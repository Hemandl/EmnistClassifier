# EMNIST-Klassifikation mit PyTorch

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)

Modulare Klassifikationspipeline für EMNIST mit Transfer-Learning und Hyperparameter-Optimierung.

## 🚀 Hauptmerkmale

- **Architektur**
  - 🧩 Multiplikative Fusion von ResNet18 und Kategorie-Klassifikator
  - 🔀 Temperatur-skalierte Wahrscheinlichkeitsfusion 
  - 🏗️ Adaptives Layer-Freezing 


- **Technologiestack**
  - 🐍 Python 3.12 + PyTorch 2.7
  - 📊 Optuna für Hyperparameter-Optimierung
  - 🖼️ Albumentations für Data Augmentation

## 📦 Installation

### Voraussetzungen
- NVIDIA GPU (empfohlen) mit CUDA 11.8
- Python 3.12

### Setup
```bash
# Virtuelle Umgebung
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# Abhängigkeiten
pip install -r requirements.txt

# Aufruf
python main.py --model base --train 
