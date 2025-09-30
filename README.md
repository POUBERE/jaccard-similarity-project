# Projet Machine Learning non Supervisé - Similarité de Jaccard

## 📋 Description

Ce projet implémente un **calculateur de similarité de Jaccard** pour comparer des phrases. La similarité de Jaccard est une métrique statistique utilisée en machine learning non supervisé pour mesurer la ressemblance entre deux ensembles.

### Formule de Jaccard

```
Jaccard(A,B) = |A ∩ B| / |A ∪ B|
```

Où :

- `A` et `B` sont les ensembles de mots des deux phrases
- `|A ∩ B|` est le nombre de mots communs (intersection)
- `|A ∪ B|` est le nombre total de mots uniques (union)

## 👥 Équipe

- OUEDRAOGO LASSINA
- OUEDRAOGO RASMANE
- POUBERE ABDOURAZAKOU

## 🚀 Installation et Prérequis

### Prérequis

- Python 3.6 ou plus récent
- Aucune dépendance externe requise (utilise uniquement la bibliothèque standard Python)

### Installation

```bash
# Récupération du projet
git clone https://github.com/POUBERE/jaccard-similarity-project.git
cd jaccard-similarity-project

# Prêt à l'emploi !
```

## 📖 Mode d'exécution

### 1. Exécution avec tests automatiques (mode par défaut)

```bash
python jaccard_similarity.py
```

Cette commande lance le programme avec des exemples de démonstration qui montrent :

- Tests de similarité entre paires de phrases
- Analyse détaillée d'un calcul
- Génération d'une matrice de similarité

### 2. Mode interactif

```bash
python jaccard_similarity.py --interactive
```

Permet de saisir vos propres phrases et de calculer leur similarité en temps réel.

**Exemple d'utilisation :**

```
Phrase 1: Le chat mange des croquettes
Phrase 2: Le chien mange des croquettes

Résultat:
  Similarité de Jaccard: 0.6667
  Mots phrase 1: ['chat', 'croquettes', 'des', 'le', 'mange']
  Mots phrase 2: ['chien', 'croquettes', 'des', 'le', 'mange']
  Mots communs: ['croquettes', 'des', 'le', 'mange'] (4)
  Mots total: ['chat', 'chien', 'croquettes', 'des', 'le', 'mange'] (6)
```
