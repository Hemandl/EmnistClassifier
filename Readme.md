<h1>EMNIST-Klassifikation mit PyTorch</h1>

Dieses Projekt implementiert einen modularen Klassifikator für den EMNIST-Datensatz unter Verwendung von PyTorch. Es demonstriert Transfer-Learning mit ResNet und eine innovative modulare Architektur zur Verbesserung der Klassifikationsgenauigkeit bei ähnlichen Zeichen.

<h2>Projektmerkmale</h2>
🚀 Transfer-Learning mit ResNet18 für EMNIST

🧩 Modulare Architektur mit multiplikativer Fusion

⚙️ Automatisierte Hyperparameter-Optimierung mit Optuna

📊 Umfassende Evaluierung mit Verwechslungsmatrizen und kategoriespezifischen Metriken



<h2>Installation</h2>
<h3>Voraussetzungen</h3>
Python 3.9

CUDA 11.7 (für GPU-Beschleunigung)

PyTorch 

<h3>Setup</h3>
<h4>Virtuelle Umgebung erstellen und aktivieren:</h4>
<code>
bash
python -m venv venv
source venv/bin/activate
</code>
<h4>Abhängigkeiten installieren:</h4>
<code>
bash
pip install -r requirements.txt
</code>
<h4>EMNIST-Datensatz herunterladen (automatisch beim ersten Ausführen)
</h4>

<h2>Schlüsseltechniken</h2>
1. Transfer-Learning: Pre-trained ResNet18 mit Feinabstimmung 

2. Hierarchische Klassifikation: Multiplikative Fusion von Teilmodulen 

3. Data Augmentation: Rotation, Translation, Perspektivenverzerrung 

4. Hyperparameter-Optimierung: TPE-Sampler mit Median-Pruning


<h2>Verwendung</h2>
<h3>Hyperparameter-Optimierung</h3>

#### Für modulares Modell
<code>
python main.py --phase tune --model_type modular --tune_trials 30
</code>

#### Für ResNet-Baseline
<code>
python main.py --phase tune --model_type resnet --tune_trials 20
</code>

### Modelltraining
#### Modulares Modell trainieren
<code>
python main.py --phase train --model_type modular --epochs 30
</code>

#### ResNet-Baseline trainieren
<code>
python main.py --phase train --model_type resnet --epochs 30
</code>

### Evaluation
#### Modulares Modell evaluieren
<code>
python main.py --phase test --model_type modular
</code>

#### ResNet-Baseline evaluieren
<code>
python main.py --phase test --model_type resnet
</code>

### Visualisierung
<code>
jupyter notebook notebooks/visualization.ipynb
</code>