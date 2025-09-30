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

### 3. Options avancées

```bash
# Prise en compte de la casse
python jaccard_similarity.py --case-sensitive

# Conservation de la ponctuation
python jaccard_similarity.py --keep-punctuation

# Combinaison possible
python jaccard_similarity.py --interactive --case-sensitive --keep-punctuation
```

### 4. Afficher l'aide

```bash
python jaccard_similarity.py --help
```

### 5. Lancer les tests unitaires

```bash
python test_jaccard.py
```

## 🧪 Exemples de tests

### Exemple 1 : Phrases similaires

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()
sentence1 = "Le chat mange des croquettes"
sentence2 = "Le chien mange des croquettes"
similarity = calculator.calculate_similarity(sentence1, sentence2)
print(f"Similarité: {similarity:.4f}")  # Résultat: 0.6667
```

**Explication :**

- Mots phrase 1 : {le, chat, mange, des, croquettes}
- Mots phrase 2 : {le, chien, mange, des, croquettes}
- Intersection : {le, mange, des, croquettes} = 4 mots
- Union : {le, chat, chien, mange, des, croquettes} = 6 mots
- Similarité : 4/6 ≈ 0.6667

### Exemple 2 : Phrases identiques

```python
sentence1 = "Python est génial"
sentence2 = "Python est génial"
similarity = calculator.calculate_similarity(sentence1, sentence2)
# Résultat: 1.0 (similarité parfaite)
```

### Exemple 3 : Aucun mot commun

```python
sentence1 = "Chat noir"
sentence2 = "Chien blanc"
similarity = calculator.calculate_similarity(sentence1, sentence2)
# Résultat: 0.0 (aucune similarité)
```
