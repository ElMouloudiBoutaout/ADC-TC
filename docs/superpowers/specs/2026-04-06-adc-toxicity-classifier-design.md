# ADC Toxicity Classifier — Design Spec
**Date:** 2026-04-06

---

## 1. Objectif

Construire un classificateur de toxicité des ADC (Antibody-Drug Conjugates) composé de :
1. Un **notebook Jupyter** pour l'exploration des données, l'entraînement et l'évaluation du modèle
2. Une **application Streamlit** pour la prédiction interactive

---

## 2. Données

**Source :** `TADC_Complete_v3_avec_S.xlsx`, feuille `RÉSUMÉ_Régression`  
**Taille :** 114 lignes × 26 colonnes, représentant 15 ADCs approuvés × plusieurs organes/effets indésirables

### Features d'entrée

| Feature | Type | Description |
|---|---|---|
| `P` | Numérique | Puissance du payload (ex: DM1, MMAE, DXd) |
| `D` | Numérique | DAR — Drug Antibody Ratio |
| `H` | Numérique | Hydrophobie (risque de capture hépatique) |
| `B` | Numérique | Effet bystander (diffusion hors cellule cible) |
| `L` | Numérique | Stabilité du linker en circulation |
| `E` | Numérique | Exposition (dose × durée) |
| `V` | Numérique | Valeur de l'organe (importance + expression on-target) |
| `S` | Numérique | Sensibilité patient (fragilité, comorbidités) |
| `Payload class` | Catégorielle | Classe du payload (Calicheamicin, MMAE, DXd, etc.) |
| `Organe` | Catégorielle | Organe affecté (Moelle, Poumon, Cornée, etc.) |

### Variables cibles

| Target | Type | Description |
|---|---|---|
| `%G≥3 observé` | Régression | Pourcentage clinique observé de toxicité Grade ≥3 |
| `Y binaire (G≥3 >10%)` | Classification binaire | 1 si G≥3 > 10%, sinon 0 |

> **Note :** `Grade prédit` et `T-ADC v3` sont des sorties de formule déterministe — ils servent de baseline de comparaison, pas de cibles ML.

---

## 3. Architecture

```
adc_toxicity_classifier/
├── data/
│   └── TADC_Complete_v3_avec_S.xlsx
├── notebook.ipynb          ← pipeline complet d'entraînement
├── models/
│   ├── model_regression.pkl
│   └── model_classification.pkl
├── app.py                  ← application Streamlit
└── requirements.txt
```

---

## 4. Pipeline ML (notebook.ipynb)

### 4.1 Chargement & nettoyage
- Lecture de la feuille `RÉSUMÉ_Régression` avec `pandas`
- Lecture avec `openpyxl` en mode `data_only=True` pour récupérer les valeurs cachées (pas les formules string). Si les valeurs cachées sont absentes, recalcul manuel des colonnes pondérées à partir de P, D, H, B, L, E, V, S
- Suppression des colonnes de formules intermédiaires (`5P`, `1D`, etc.)
- Encodage one-hot de `Payload class` et `Organe`

### 4.2 Validation
Utiliser **Leave-One-ADC-Out cross-validation** (LOAOCV) :
- 15 folds, chaque fold exclut toutes les lignes d'un ADC
- Évite le data leakage entre lignes du même ADC
- Représente fidèlement la capacité de généralisation sur un ADC inconnu

### 4.3 Modèles entraînés
- **Random Forest** (régression + classification)
- **XGBoost** (régression + classification)
- Hyperparamètres optimisés par GridSearchCV dans chaque fold LOAOCV

### 4.4 Baseline
Comparer les modèles ML au score `T-ADC v3` (formule) sur les mêmes métriques.

### 4.5 Métriques d'évaluation

| Tâche | Métriques |
|---|---|
| Régression | MAE, RMSE, R² |
| Classification | AUC-ROC, F1-score, Accuracy |

### 4.6 Interprétabilité
- SHAP values pour les deux modèles (summary plot + bar plot)
- Identifier les features les plus importantes pour la prédiction de toxicité

### 4.7 Export
- Meilleur modèle de régression → `models/model_regression.pkl`
- Meilleur modèle de classification → `models/model_classification.pkl`
- Encodeurs et scalers → `models/preprocessor.pkl`

---

## 5. Application Streamlit (app.py)

### Interface
- **Sidebar :** sliders pour chaque feature numérique (P, D, H, B, L, E, V, S) avec valeurs min/max extraites du dataset
- **Dropdowns :** sélection `Payload class` et `Organe`
- **Bouton :** "Prédire la toxicité"

### Résultats affichés
- Score `%G≥3 prédit` (régression) avec jauge visuelle
- Grade de risque (classification binaire) avec badge coloré : vert (faible) / rouge (élevé)
- Score T-ADC v3 calculé par formule (baseline de comparaison)
- SHAP waterfall plot pour l'entrée courante

---

## 6. Dépendances (requirements.txt)

```
pandas
openpyxl
scikit-learn
xgboost
shap
streamlit
joblib
matplotlib
```

---

## 7. Contraintes & limites connues

- **Dataset très petit (114 lignes / 15 ADCs)** : le modèle ML peut ne pas surpasser la formule T-ADC v3 — c'est une conclusion valide et attendue
- Les features `P, D, H, B, L, E` sont des **scores experts** (non mesurés directement), ce qui introduit un biais de saisie
- La LOAOCV donne des intervalles de confiance larges avec 15 folds seulement

---

## 8. Hors scope

- Application web avec backend séparé (FastAPI, etc.)
- Déploiement cloud
- Ingestion de nouveaux ADC en temps réel
- Support des ADC bispécifiques
