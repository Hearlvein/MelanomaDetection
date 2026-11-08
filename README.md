# Détection du mélanome sur HAM10000

Projet scolaire de classification binaire : distinguer les images de **mélanome**
(`dx = mel`) des autres diagnostics dermatoscopiques de HAM10000.

Le jeu de données contient 10 015 images provenant de 7 470 lésions. La classe
positive est rare (11,1 %), donc l'exactitude est trompeuse. Les métriques
principales sont la précision moyenne (AP), le ROC-AUC, la sensibilité, la
spécificité et le F2.

## Méthode en quatre étapes

1. **`01_dataset.py`** — fusionne le classeur de caractéristiques avec les
   métadonnées officielles, définit la cible mélanome et crée un découpage
   60/20/20. Toutes les images d'une même lésion restent dans le même ensemble
   (`StratifiedGroupKFold`, graine 42).
2. **`02_compare_models.py`** — évalue des références simples et quatre familles
   de modèles avec plusieurs stratégies de déséquilibre. Les hyperparamètres sont
   réglés par validation croisée groupée **dans l'ensemble d'entraînement**.
   Le seuil de décision est choisi sur les prédictions hors-pli, puis gelé.
3. **`03_final_test.py`** — évalue le modèle gelé sur l'ensemble de test **une
   seule fois**. Le script refuse d'écraser un résultat existant sans `--force`.
4. **`04_deep_learning.py`** — vérifie les images, entraîne un réseau
   convolutif (`custom_cnn`) ou un VGG16 transféré (`vgg16`), puis permet
   l'évaluation unique sur le test.

L'ensemble de test ne sert jamais à l'entraînement, au réglage, à la sélection de
variables ou au choix du seuil. Les intervalles de confiance rééchantillonnent
des lésions entières, pas des images isolées.

## Données et variables

Entrées brutes :

- `data/raw/HAM10000_metadata.csv`
- `data/raw/HAM10000_features_couleurs.xlsx`

Les modèles tabulaires utilisent l'âge, le sexe, la localisation, 12
caractéristiques de couleur, 2 d'asymétrie et 2 de forme. Les identifiants et
les diagnostics ne sont jamais des variables explicatives.

| Ensemble | Images | Lésions | Mélanome | Autre | Prévalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| entraînement | 6 009 | 4 482 | 668 | 5 341 | 0,111 |
| validation | 2 003 | 1 494 | 222 | 1 781 | 0,111 |
| test | 2 003 | 1 494 | 223 | 1 780 | 0,111 |

Empreinte du découpage :
`b39eae73a410580368397df3c203b644d7887084cea27627084a318a5cffa862`.

## Modèles comparés

Références : classe majoritaire, scores aléatoires, puis régression logistique
limitée à l'âge, aux métadonnées, à la couleur, à la forme ou à toutes les
variables.

Modèles principaux :

- régression logistique ;
- forêt aléatoire ;
- HistGradientBoosting ;
- ExtraTrees.

Chaque famille est testée sans correction du déséquilibre, avec des poids de
classe équilibrés et avec SMOTE (ou sous-échantillonnage aléatoire lorsque la
validation croisée le recommande). Le pré-traitement, la sélection de variables
et le rééchantillonnage sont ajustés à l'intérieur de chaque pli.

Modèles profonds :

- `custom_cnn` : petit réseau convolutif entraîné depuis zéro ;
- `vgg16` : base VGG16 ImageNet gelée avec une petite tête de classification.

## Résultats

Modèle gelé : **ExtraTrees** avec `class_weight="balanced"`, 300 arbres,
profondeur 30, `min_samples_leaf=4`, `SelectKBest(k=30)` et un seuil F2 de
**0,220**.

Validation : AP 0,359, ROC-AUC 0,820, F2 0,533.

Test (une seule évaluation) :

| Métrique | Valeur (IC 95 %) |
| --- | --- |
| Sensibilité | 0,834 [0,784–0,884] |
| Spécificité | 0,696 [0,671–0,721] |
| VPP | 0,256 [0,212–0,299] |
| VPN | 0,971 [0,960–0,981] |
| F1 | 0,392 [0,336–0,441] |
| F2 | 0,574 [0,517–0,622] |
| Précision moyenne | 0,391 [0,311–0,470] |
| ROC-AUC | 0,835 [0,806–0,862] |
| Score de Brier | 0,092 [0,084–0,101] |

Matrice de confusion au seuil 0,220 : TN 1 239, FP 541, FN 37, TP 186.

Un second seuil à 0,134 vise une sensibilité de 0,90 : sensibilité 0,928,
spécificité 0,518, VPP 0,194, VPN 0,983.

Le rapport consolidé, les tableaux détaillés et toutes les métriques sont dans
`reports/RESULTS.md` et `reports/metrics/`. Les quatre figures du test sont dans
`reports/figures/final_test_melanoma_*.png`.

## Reproduire

```bash
pip install -e ".[tabular,dev]"

python scripts/01_dataset.py
python scripts/02_compare_models.py
```

Le test a déjà été évalué une fois. **Ne pas relancer** `03_final_test.py` :
le script s'arrête si `reports/metrics/final_test.json` existe.

Pour rejouer l'évaluation une seule fois et volontairement :

```bash
python scripts/03_final_test.py --force
```

Modèles profonds (nécessite TensorFlow et les images HAM10000) :

```bash
pip install -e ".[tabular,deep,dev]"

python scripts/04_deep_learning.py check-images
python scripts/04_deep_learning.py train --model custom_cnn --epochs 20
python scripts/04_deep_learning.py train --model vgg16 --epochs 20
```

Tests :

```bash
pytest -q
```

## Limites

- HAM10000 est un jeu dermatoscopique ; les modèles tabulaires utilisent des
  caractéristiques déjà extraites, pas les pixels bruts.
- Aucun identifiant patient n'est disponible. Le regroupement par lésion est le
  meilleur contrôle de fuite disponible, mais il reste imparfait.
- Le jeu n'est pas représentatif de toutes les carnations, populations ou
  pratiques cliniques.
- La validation a servi à sélectionner le modèle : son estimation est
  légèrement optimiste. C'est l'estimation sur le test qu'il faut rapporter.
- La calibration est imparfaite, surtout après pondération des classes.
- Ce projet n'est pas un dispositif médical et ne doit jamais guider un soin.
