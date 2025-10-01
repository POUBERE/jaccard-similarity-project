# Guide de Démarrage Rapide

## Installation en 30 secondes

```bash
# Cloner le dépôt
git clone https://github.com/[votre-username]/jaccard-similarity-project.git
cd jaccard-similarity-project

# Aucune installation requise, le projet est prêt à l'emploi
python jaccard_similarity.py
```

## Utilisation basique

### 1. Tests automatiques

```bash
python jaccard_similarity.py
```

Lance des exemples de démonstration.

### 2. Mode interactif

```bash
python jaccard_similarity.py --interactive
```

Entrez ensuite vos phrases :

```
Phrase 1: Le chat mange
Phrase 2: Le chien mange
```

### 3. Utiliser comme module Python

```python
from jaccard_similarity import JaccardSimilarity

# Initialisation du calculateur
calc = JaccardSimilarity()

# Calcul de la similarité
sim = calc.calculate_similarity("phrase 1", "phrase 2")
print(f"Similarité: {sim:.4f}")
```

## Exemples rapides

### Exemple 1 : Comparaison simple

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity()
result = calc.calculate_similarity(
    "Le chat mange des croquettes",
    "Le chien mange des croquettes"
)
print(result)  # 0.6667
```

### Exemple 2 : Analyse détaillée

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity()
details = calc.calculate_similarity_detailed(
    "Le chat mange",
    "Le chien mange"
)

print(f"Similarité: {details['jaccard_similarity']}")
print(f"Mots communs: {details['intersection']}")
print(f"Tous les mots: {details['union']}")
```

### Exemple 3 : Comparer plusieurs phrases

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity()
phrases = [
    "Le chat mange",
    "Le chien mange",
    "Python est génial"
]

# Recherche de la paire la plus similaire
idx1, idx2, sim = calc.get_most_similar_pair(phrases)
print(f"Phrases {idx1+1} et {idx2+1} sont les plus similaires: {sim:.4f}")
```

### Exemple 4 : Matrice de similarité

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity()
phrases = ["chat", "chien", "oiseau"]

matrix = calc.get_similarity_matrix(phrases)
for row in matrix:
    print([f"{val:.2f}" for val in row])
```

## Options de configuration

```python
# Prendre en compte la casse
calc = JaccardSimilarity(case_sensitive=True)

# Conserver la ponctuation
calc = JaccardSimilarity(remove_punctuation=False)

# Combiner plusieurs options
calc = JaccardSimilarity(case_sensitive=True, remove_punctuation=False)
```

## Lancer les tests

```bash
python test_jaccard.py
```

## Lancer la démonstration avancée

```bash
python examples/demo.py
```

## Cas d'usage pratiques

### Détection de plagiat

```python
calc = JaccardSimilarity()
original = "L'intelligence artificielle transforme notre société"
suspect = "L'IA transforme notre société moderne"

sim = calc.calculate_similarity(original, suspect)
if sim > 0.5:
    print("⚠️  Suspicion de plagiat")
```

### Moteur de recherche simple

```python
calc = JaccardSimilarity()
documents = [
    "Python est un langage de programmation",
    "Java est utilisé en entreprise",
    "Le chat dort sur le canapé"
]

query = "langage programmation"
scores = [(i, calc.calculate_similarity(query, doc))
          for i, doc in enumerate(documents)]
scores.sort(key=lambda x: x[1], reverse=True)

print("Résultats de recherche:")
for idx, score in scores[:3]:
    if score > 0:
        print(f"  {documents[idx]} (score: {score:.2f})")
```

## Aide et support

```bash
# Affichage de l'aide
python jaccard_similarity.py --help

# Consultation du README complet
cat README.md
```

## Prochaines étapes

1. Lisez le `README.md` complet pour plus de détails
2. Explorez `examples/demo.py` pour voir des applications avancées
3. Consultez `test_jaccard.py` pour comprendre les tests
4. Adaptez le code à vos besoins spécifiques

## Questions fréquentes

**Q: Quelle version de Python est requise ?**  
R: Python 3.6 ou plus récent.

**Q: Y a-t-il des dépendances externes ?**  
R: Non, le projet utilise uniquement la bibliothèque standard Python.

**Q: Comment interpréter les résultats ?**  
R:

- 1.0 = phrases identiques
- 0.8-0.99 = très similaires
- 0.5-0.79 = moyennement similaires
- 0.1-0.49 = peu similaires
- 0.0 = aucune similarité

**Q: Puis-je utiliser ce code dans mes projets ?**  
R: Oui, consultez le fichier LICENSE pour les détails.

---

Pour plus d'informations, consultez le README.md complet.
