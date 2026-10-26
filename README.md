# Lab 2 — Deep Learning (PyTorch, GPU) : CNN, Faster R-CNN, Fine-tuning, ViT sur MNIST

**FST Tanger — Cycle LSI — Pr. ELAACHAK Lotfi**
**Étudiant :** _(nom, prénom)_

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Lab2_DeepLearning.ipynb` | Notebook complet (Google Colab, GPU) |
| `resultats_lab2.csv` / `.json` | Résultats générés par le notebook |
| `README.md` | Ce rapport |

**Exécution :** ouvrir le notebook sur Google Colab → *Runtime → Change runtime type → GPU* → *Run all*.
MNIST est téléchargé via `torchvision` (mêmes données que le dataset Kaggle `hojjatk/mnist-dataset`).

## Méthodologie

**1. CNN.** 2 blocs `Conv 3×3 (padding 1, stride 1) → BatchNorm → ReLU → MaxPool 2×2` (32 puis 64 filtres), puis `Dropout(0.5) → FC 128 → FC 10`.
Adam (lr 1e-3), weight decay 1e-4 (régularisation L2), `StepLR`, batch 128, 5 époques, jeu complet (60 000 / 10 000).

**2. Faster R-CNN.** MNIST n'a pas de boîtes englobantes : elles sont calculées automatiquement (boîte des pixels > seuil) sur des images 64×64 où le chiffre est placé aléatoirement.
Modèle `fasterrcnn_mobilenet_v3_large_320_fpn` (11 classes = 10 chiffres + fond), backbone pré-entraîné ImageNet, SGD, sous-ensemble de 10 000 images d'entraînement / 2 000 de test.
La classe prédite d'une image est celle de la détection au meilleur score.

**3. Fine-tuning.** VGG16 et AlexNet (poids ImageNet), images 64×64 en 3 canaux, partie convolutive gelée, dernière couche remplacée par `Linear(4096, 10)`, Adam (lr 1e-4), 2 époques,
sous-ensemble de 20 000 images d'entraînement / 5 000 de test.

**4. ViT from scratch.** Patchs 4×4 (49 patchs), embedding linéaire, token `[CLS]`, encodage positionnel sinusoïdal, 4 blocs Transformer (4 têtes, `hidden_d = 64`),
attention multi-têtes écrite à la main, AdamW (lr 1e-3), scheduler cosinus, 10 époques, jeu complet.

## Résultats (exécution sur Google Colab, GPU)

| Modèle | Accuracy | F1 (macro) | Loss (test) | Temps d'entraînement (s) | Paramètres entraînables |
|---|---|---|---|---|---|
| CNN | **0.9929** | **0.9929** | **0.0225** | 80.2 | 421 834 |
| ViT (scratch) | 0.9849 | 0.9848 | 0.0495 | 183.0 | 201 866 |
| VGG16 (fine-tuned) | 0.9644 | 0.9644 | 0.1122 | 55.2 | 119 586 826 |
| AlexNet (fine-tuned) | 0.9344 | 0.9341 | 0.2048 | **25.4** | 54 575 114 |
| Faster R-CNN | — | — | NaN | — | 18 917 458 |

**Faster R-CNN — première exécution : échec.** La loss est devenue `NaN` dès la 1ʳᵉ époque (SGD, lr = 0.01), le modèle n'a donc produit aucune détection (accuracy et F1 à 0, voir `Faster R-CNN — Loss` vide).
Ce n'est pas un résultat de performance mais une divergence de l'entraînement ; le temps de 305 s n'est donc pas comparable. Le notebook a été corrigé (lr = 0.002, *gradient clipping*, saut des batchs non finis) :
**relancer la partie 1.2 et remplacer la ligne Faster R-CNN ci-dessus par les nouvelles valeurs** avant de rendre le rapport.

Précautions de lecture :
- Les trois modèles CNN / ViT utilisent le jeu complet ; VGG16 et AlexNet sont entraînés sur 20 000 images (2 époques) et testés sur 5 000 images 64×64 ; Faster R-CNN sur 10 000 / 2 000 images 64×64. La comparaison n'est donc pas strictement à données identiques.
- La loss de Faster R-CNN (somme de 4 termes) n'est pas comparable à la cross-entropy des autres modèles.
- Les temps dépendent du GPU alloué par Colab et du nombre d'époques (5 pour le CNN, 10 pour le ViT, 2 pour VGG16/AlexNet).

Courbes : CNN — la loss d'entraînement et de test diminuent, l'accuracy passe de 0.9786 à 0.9929 en 5 époques (légère baisse à l'époque 4). ViT — l'accuracy passe de 0.9467 à 0.9849 en 10 époques, la loss de test continue de baisser (0.1749 → 0.0495). _(Insérer ici les images exportées du notebook.)_

## Analyse

**CNN.** C'est le meilleur modèle de ce lab : 99,29 % d'accuracy avec seulement 421 834 paramètres et 80 s d'entraînement. Les convolutions (filtres locaux partagés) et le pooling sont bien adaptés à des petites images de chiffres. Les courbes train/test restent proches, signe d'un faible surapprentissage (BatchNorm, Dropout, weight decay).

**ViT from scratch.** Il atteint 98,49 %, soit environ 0,8 point de moins que le CNN, avec deux fois moins de paramètres (201 866) mais un entraînement plus de deux fois plus long (183 s contre 80 s, avec 10 époques contre 5). Le ViT n'a pas le biais inductif de localité des convolutions : il doit apprendre la structure spatiale à partir des données, ce qui explique sa convergence plus lente (accuracy encore croissante à la dernière époque). Sur un jeu simple comme MNIST, le CNN garde l'avantage ; l'avantage des transformers apparaît plutôt avec de grands jeux de données et des images complexes.

**Fine-tuning VGG16 / AlexNet.** Ils restent derrière le CNN (96,44 % et 93,44 %) alors qu'ils ont 100 à 300 fois plus de paramètres entraînables. Plusieurs facteurs possibles : les filtres ImageNet (images naturelles en couleur) sont mal adaptés à des chiffres en niveaux de gris ; la partie convolutive est gelée ; l'entraînement est limité à 2 époques sur 20 000 images. VGG16, plus profond, fait mieux qu'AlexNet (+3 points) mais coûte plus de temps (55 s contre 25 s). Ces explications sont des hypothèses non testées ici (il faudrait par exemple dégeler le dernier bloc convolutif avec `UNFREEZE_LAST_BLOCK = True` ou entraîner plus longtemps). Conclusion : sur MNIST, le transfert d'apprentissage n'apporte pas de gain par rapport à un petit CNN entraîné de zéro, mais il permet d'obtenir ~94–96 % en très peu d'époques.

**Faster R-CNN.** Première exécution inexploitable (loss NaN). Il faut relancer avec le notebook corrigé et compléter cette section avec les nouveaux résultats. À noter, quel que soit le résultat : Faster R-CNN est un modèle de détection (RPN + têtes RoI), beaucoup plus lourd et plus lent qu'un classifieur, et MNIST n'a pas de vraies boîtes, donc ce modèle n'a pas d'intérêt pratique ici par rapport au CNN.

**Conclusion générale.** Pour MNIST, un CNN léger offre le meilleur compromis précision / coût (99,29 %), suivi du ViT (98,49 %), puis des modèles fine-tunés. Le choix de l'architecture dépend de la tâche : classification simple → CNN ; données volumineuses et complexes → ViT ; peu de données d'images naturelles → transfert d'apprentissage.

## Synthèse personnelle

Pendant ce lab, j'ai appris à construire et entraîner des réseaux de neurones avec PyTorch en mode GPU : boucle d'entraînement, `DataLoader`, évaluation et choix des hyperparamètres (noyaux, padding, stride, optimiseur, régularisation). J'ai mesuré les modèles avec plusieurs métriques (accuracy, F1, loss, temps, nombre de paramètres) et constaté qu'un petit CNN bien réglé peut surpasser des modèles beaucoup plus gros sur MNIST. J'ai compris la différence entre classification et détection d'objets en adaptant Faster R-CNN, et j'ai rencontré un cas concret de divergence d'entraînement (loss `NaN`) qu'on corrige en réduisant le learning rate et en limitant les gradients. J'ai vu l'intérêt et les limites du transfert d'apprentissage avec VGG16 et AlexNet, et j'ai implémenté un Vision Transformer à la main (patchs, token `[CLS]`, attention multi-têtes), ce qui m'a montré qu'il apprend plus lentement qu'un CNN sur un petit jeu de données.
