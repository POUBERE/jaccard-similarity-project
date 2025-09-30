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

### Exemple 4 : Analyse détaillée

```python
result = calculator.calculate_similarity_detailed("Le chat mange", "Le chien mange")
print(result)
# {
#   'sentence1': 'Le chat mange',
#   'sentence2': 'Le chien mange',
#   'words_set1': {'le', 'chat', 'mange'},
#   'words_set2': {'le', 'chien', 'mange'},
#   'intersection': {'le', 'mange'},
#   'union': {'le', 'chat', 'chien', 'mange'},
#   'intersection_size': 2,
#   'union_size': 4,
#   'jaccard_similarity': 0.5
# }
```

### Exemple 5 : Comparaison multiple

```python
sentences = [
    "Le chat mange",
    "Le chien mange",
    "Les animaux mangent",
    "Python est génial"
]

# Comparaison de toutes les paires
results = calculator.compare_multiple_sentences(sentences)
for idx1, idx2, sim in results:
    print(f"Phrases {idx1+1} et {idx2+1}: {sim:.4f}")

# Recherche de la paire la plus similaire
idx1, idx2, max_sim = calculator.get_most_similar_pair(sentences)
print(f"Paire la plus similaire: phrases {idx1+1} et {idx2+1} ({max_sim:.4f})")
```

### Exemple 6 : Matrice de similarité

```python
matrix = calculator.get_similarity_matrix(sentences)
# Retourne une matrice n×n avec les similarités entre toutes les phrases
```

## 🔧 Fonctionnalités

### Classe `JaccardSimilarity`

#### Paramètres de configuration

- **`case_sensitive`** (bool, défaut=False) : Si True, respecte la casse des mots
- **`remove_punctuation`** (bool, défaut=True) : Si True, supprime la ponctuation

#### Méthodes principales

1. **`calculate_similarity(sentence1, sentence2)`**

      - Calcule la similarité de Jaccard entre deux phrases
      - Retourne une valeur entre 0 (aucune similarité) et 1 (identiques)

2. **`calculate_similarity_detailed(sentence1, sentence2)`**

      - Version détaillée avec toutes les informations du calcul
      - Retourne un dictionnaire avec les ensembles, intersection, union, etc.

3. **`compare_multiple_sentences(sentences)`**

      - Compare toutes les paires dans une liste de phrases
      - Retourne une liste de tuples (index1, index2, similarité)

4. **`get_similarity_matrix(sentences)`**

      - Génère une matrice de similarité n×n
      - Utile pour visualiser toutes les relations

5. **`get_most_similar_pair(sentences)`**

      - Trouve la paire la plus similaire dans une liste
      - Retourne (index1, index2, similarité_max)

6. **`preprocess_sentence(sentence)`**
      - Prétraite une phrase (conversion en ensemble de mots)
      - Applique les transformations selon la configuration

### Prétraitement automatique

Le programme applique automatiquement les transformations suivantes :

- ✅ Conversion en minuscules (si case_sensitive=False)
- ✅ Suppression de la ponctuation (si remove_punctuation=True)
- ✅ Gestion des accents français et caractères spéciaux
- ✅ Division en mots individuels
- ✅ Suppression des espaces multiples
- ✅ Élimination des mots vides (chaînes vides)

## 📊 Résultats d'exemple

| Phrase 1                       | Phrase 2                              | Similarité | Interprétation        |
| ------------------------------ | ------------------------------------- | ---------- | --------------------- |
| "Le chat mange"                | "Le chien mange"                      | 0.5000     | Moyennement similaire |
| "Python est un langage"        | "Java est un langage"                 | 0.7500     | Très similaire        |
| "Bonjour monde"                | "Hello world"                         | 0.0000     | Aucune similarité     |
| "Machine learning supervisé"   | "Apprentissage automatique supervisé" | 0.2500     | Faible similarité     |
| "Le chat mange des croquettes" | "Le chien mange des croquettes"       | 0.6667     | Assez similaire       |
