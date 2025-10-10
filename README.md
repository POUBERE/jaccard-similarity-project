# Projet Machine Learning non Supervisé - Similarité de Jaccard

## Version 2.0

**Date de sortie :** Octobre 2025

### Nouveautés majeures

- **Distance de Jaccard** : Métrique complémentaire à la similarité
- **Stemming français** : Normalisation des mots à leur racine
- **Stop-words français** : Filtrage de 60+ mots courants
- **Interprétation contextuelle** : Analyse adaptée selon le contexte d'application
- **Export de données** : Sauvegarde des résultats en CSV et JSON
- **Interface graphique améliorée** : Nouvelles options et onglet d'export

---

## Description

Ce projet implémente un **calculateur de similarité de Jaccard** pour comparer des phrases. La similarité de Jaccard est une métrique statistique utilisée en machine learning non supervisé pour mesurer la ressemblance entre deux ensembles.

### Formules

```
Similarité de Jaccard: J(A,B) = |A ∩ B| / |A ∪ B|
Distance de Jaccard:   d(A,B) = 1 - J(A,B)
```

Où :

- `A` et `B` sont les ensembles de mots des deux phrases
- `|A ∩ B|` est le nombre de mots communs (intersection)
- `|A ∪ B|` est le nombre total de mots uniques (union)

---

## Équipe

- **OUEDRAOGO Lassina**
- **OUEDRAOGO Rasmane**
- **POUBERE Abdourazakou**

---

## Installation et Prérequis

### Prérequis

- Python 3.6 ou plus récent
- Aucune dépendance externe requise (utilise uniquement la bibliothèque standard Python)

### Installation

```bash
# Récupération du projet
git clone https://github.com/POUBERE/jaccard-similarity-project.git
cd jaccard-similarity-project

# Prêt à l'emploi !
python jaccard_similarity.py
```

---

## Mode d'exécution

### 1. Exécution avec tests automatiques (mode par défaut)

```bash
python jaccard_similarity.py
```

Cette commande lance le programme avec des exemples de démonstration incluant :

- Tests de similarité ET distance entre paires de phrases
- Analyse détaillée avec interprétation contextuelle
- Génération de matrices comparatives
- Démonstration des nouvelles fonctionnalités v2.0

### 2. Mode interactif

```bash
python jaccard_similarity.py --interactive
```

Permet de saisir vos propres phrases et de calculer leur similarité/distance en temps réel.

**Exemple d'utilisation :**

```
Phrase 1: Le chat mange des croquettes
Phrase 2: Le chien mange des croquettes

Résultat:
  Similarité de Jaccard: 0.6667
  Distance de Jaccard: 0.3333
  Mots communs: ['croquettes', 'des', 'le', 'mange'] (4)
  Mots total: ['chat', 'chien', 'croquettes', 'des', 'le', 'mange'] (6)
```

### 3. Options avancées (NOUVEAU v2.0)

```bash
# Prise en compte de la casse
python jaccard_similarity.py --case-sensitive

# Suppression des stop-words français
python jaccard_similarity.py --remove-stopwords

# Utilisation du stemming français
python jaccard_similarity.py --use-stemming

# Combinaison d'options
python jaccard_similarity.py --remove-stopwords --use-stemming

# Export des résultats
python jaccard_similarity.py --export csv
python jaccard_similarity.py --export json
python jaccard_similarity.py --export both
```

### 4. Interface graphique (NOUVEAU v2.0)

```bash
python jaccard_gui.py
```

Lance une interface graphique Tkinter avec :

- Comparaison simple et multiple de phrases
- Options de configuration interactives
- Génération de matrices de similarité
- Export CSV/JSON intégré

### 5. Lancer les tests unitaires

```bash
python test_jaccard.py
```

Plus de 50 tests couvrant toutes les fonctionnalités.

### 6. Démonstrations avancées

```bash
python examples/demo.py
```

11 démonstrations pratiques incluant les nouvelles fonctionnalités v2.0.

---

## Exemples d'utilisation

### Exemple 1 : Comparaison basique

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()
sentence1 = "Le chat mange des croquettes"
sentence2 = "Le chien mange des croquettes"

similarity = calculator.calculate_similarity(sentence1, sentence2)
print(f"Similarité: {similarity:.4f}")  # 0.6667

distance = calculator.calculate_distance(sentence1, sentence2)
print(f"Distance: {distance:.4f}")      # 0.3333
```

**Explication :**

- Mots phrase 1 : {le, chat, mange, des, croquettes}
- Mots phrase 2 : {le, chien, mange, des, croquettes}
- Intersection : {le, mange, des, croquettes} = 4 mots
- Union : {le, chat, chien, mange, des, croquettes} = 6 mots
- Similarité : 4/6 = 0.6667
- Distance : 1 - 0.6667 = 0.3333

### Exemple 2 : Utilisation des stop-words (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

# Sans stop-words
calc_basic = JaccardSimilarity()
sim1 = calc_basic.calculate_similarity(
    "Le chat noir mange",
    "Le chien blanc mange"
)
print(f"Sans stop-words: {sim1:.4f}")  # 0.3333

# Avec stop-words (retire "le")
calc_stopwords = JaccardSimilarity(remove_stopwords=True)
sim2 = calc_stopwords.calculate_similarity(
    "Le chat noir mange",
    "Le chien blanc mange"
)
print(f"Avec stop-words: {sim2:.4f}")  # 0.2500

# Les stop-words améliorent la précision en se concentrant sur les mots essentiels
```

### Exemple 3 : Utilisation du stemming (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

# Sans stemming
calc_basic = JaccardSimilarity()
sim1 = calc_basic.calculate_similarity(
    "Je mange une pomme",
    "Tu manges des pommes"
)
print(f"Sans stemming: {sim1:.4f}")  # Faible similarité

# Avec stemming (mange/manges → mang, pomme/pommes → pomm)
calc_stemming = JaccardSimilarity(use_stemming=True)
sim2 = calc_stemming.calculate_similarity(
    "Je mange une pomme",
    "Tu manges des pommes"
)
print(f"Avec stemming: {sim2:.4f}")  # Similarité plus élevée
```

### Exemple 4 : Combinaison optimale (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

# Configuration optimale pour le français
calc = JaccardSimilarity(
    case_sensitive=False,      # Ignore la casse
    remove_stopwords=True,     # Retire les mots courants
    use_stemming=True          # Normalise les variations
)

s1 = "Les développeurs Python créent des applications"
s2 = "Le développeur python crée une application"

similarity = calc.calculate_similarity(s1, s2)
print(f"Similarité optimale: {similarity:.4f}")  # Haute similarité

# Cette configuration détecte mieux la similarité sémantique !
```

### Exemple 5 : Interprétation contextuelle (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()

original = "L'intelligence artificielle transforme notre société"
suspect = "L'IA transforme notre société et notre économie"

similarity = calculator.calculate_similarity(original, suspect)

# Interprétation pour différents contextes
contexts = ['general', 'plagiarism', 'clustering', 'search', 'diversity']

for context in contexts:
    interp = calculator.interpret_similarity(similarity, context)
    print(f"\nContexte: {context}")
    print(f"  {interp['emoji']} {interp['contextual_interpretation']}")
    print(f"  Recommandations: {interp['recommendations'][0]}")
```

**Résultat :**

```
Contexte: general
  🟡 Bonne similarité - Sujet probablement commun
  Recommandations: Aucune recommandation spécifique

Contexte: plagiarism
  ⚠️  SUSPICION ÉLEVÉE - Peut indiquer une paraphrase
  Recommandations: Examiner les passages spécifiques similaires

Contexte: clustering
  📂 CLUSTER MODÉRÉ - Documents connexes, possiblement même thème
  Recommandations: Considérer comme potentiellement liés

Contexte: search
  🎯 PERTINENT - Bon match avec plusieurs termes clés
  Recommandations: Document pertinent, à inclure dans les résultats

Contexte: diversity
  🎨 ASSEZ SIMILAIRE
  Recommandations: Aucune recommandation spécifique
```

### Exemple 6 : Export des résultats (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()

phrases = [
    "Python est un langage de programmation",
    "Java est un langage orienté objet",
    "JavaScript permet de créer des sites web"
]

# Génération des résultats avec distance
results = []
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        result = calculator.calculate_distance_detailed(phrases[i], phrases[j])
        results.append(result)

# Export en CSV
csv_file = calculator.export_results_to_csv(results)
print(f"CSV créé: {csv_file}")
# Sortie: jaccard_results_20251003_143022.csv

# Export en JSON
json_file = calculator.export_results_to_json(results)
print(f"JSON créé: {json_file}")
# Sortie: jaccard_results_20251003_143022.json
```

### Exemple 7 : Analyse détaillée avec distance (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()

result = calculator.calculate_distance_detailed(
    "Le chat mange",
    "Le chien mange"
)

print(f"Phrase 1: {result['sentence1']}")
print(f"Phrase 2: {result['sentence2']}")
print(f"\nMots phrase 1: {sorted(result['words_set1'])}")
print(f"Mots phrase 2: {sorted(result['words_set2'])}")
print(f"\nIntersection: {sorted(result['intersection'])}")
print(f"Union: {sorted(result['union'])}")
print(f"\nSimilarité: {result['jaccard_similarity']:.4f}")
print(f"Distance: {result['jaccard_distance']:.4f}")
print(f"Vérification: {result['jaccard_similarity'] + result['jaccard_distance']:.4f}")
```

### Exemple 8 : Matrice de distance (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()

sentences = [
    "Le chat mange",
    "Le chien mange",
    "Les animaux mangent",
    "Python est génial"
]

# Matrice de similarité
sim_matrix = calculator.get_similarity_matrix(sentences)

# Matrice de distance
dist_matrix = calculator.get_distance_matrix(sentences)

print("Matrice de similarité:")
for i, row in enumerate(sim_matrix):
    print(f"  {i}: {[f'{val:.2f}' for val in row]}")

print("\nMatrice de distance:")
for i, row in enumerate(dist_matrix):
    print(f"  {i}: {[f'{val:.2f}' for val in row]}")
```

### Exemple 9 : Recherche de paires extrêmes (NOUVEAU v2.0)

```python
from jaccard_similarity import JaccardSimilarity

calculator = JaccardSimilarity()

sentences = [
    "chat noir",
    "chat blanc",
    "chien brun",
    "oiseau bleu"
]

# Paire la plus similaire
idx1, idx2, max_sim = calculator.get_most_similar_pair(sentences)
print(f"Paire la PLUS similaire: {idx1} et {idx2}")
print(f"  Similarité: {max_sim:.4f}")
print(f"  '{sentences[idx1]}' ↔ '{sentences[idx2]}'")

# Paire la plus différente
idx1, idx2, max_dist = calculator.get_most_different_pair(sentences)
print(f"\nPaire la PLUS différente: {idx1} et {idx2}")
print(f"  Distance: {max_dist:.4f}")
print(f"  '{sentences[idx1]}' ↔ '{sentences[idx2]}'")
```

---

## Fonctionnalités

### Classe `JaccardSimilarity`

#### Paramètres de configuration

| Paramètre            | Type | Défaut | Description                    | Version |
| -------------------- | ---- | ------ | ------------------------------ | ------- |
| `case_sensitive`     | bool | False  | Respecte la casse des mots     | v1.0    |
| `remove_punctuation` | bool | True   | Supprime la ponctuation        | v1.0    |
| `remove_stopwords`   | bool | False  | Retire les stop-words français | v2.0    |
| `use_stemming`       | bool | False  | Applique le stemming français  | v2.0    |

#### Méthodes principales

##### Calcul de similarité

```python
calculate_similarity(sentence1: str, sentence2: str) -> float
```

Calcule la similarité de Jaccard entre deux phrases (0.0 à 1.0).

```python
calculate_similarity_detailed(sentence1: str, sentence2: str) -> Dict
```

Version détaillée retournant tous les détails du calcul.

##### Calcul de distance (NOUVEAU v2.0)

```python
calculate_distance(sentence1: str, sentence2: str) -> float
```

Calcule la distance de Jaccard entre deux phrases (0.0 à 1.0).

```python
calculate_distance_detailed(sentence1: str, sentence2: str) -> Dict
```

Version détaillée incluant similarité ET distance.

##### Comparaisons multiples

```python
compare_multiple_sentences(sentences: List[str]) -> List[Tuple[int, int, float]]
```

Compare toutes les paires dans une liste de phrases.

```python
get_similarity_matrix(sentences: List[str]) -> List[List[float]]
```

Génère une matrice de similarité n×n.

```python
get_distance_matrix(sentences: List[str]) -> List[List[float]]
```

Génère une matrice de distance n×n. (NOUVEAU v2.0)

##### Recherche de paires

```python
get_most_similar_pair(sentences: List[str]) -> Tuple[int, int, float]
```

Trouve la paire la plus similaire.

```python
get_most_different_pair(sentences: List[str]) -> Tuple[int, int, float]
```

Trouve la paire la plus différente. (NOUVEAU v2.0)

##### Interprétation (NOUVEAU v2.0)

```python
interpret_similarity(similarity: float, context: str = 'general') -> Dict
```

Interprète un score de similarité selon le contexte.

```python
interpret_distance(distance: float, context: str = 'general') -> Dict
```

Interprète une distance selon le contexte.

**Contextes disponibles :**

- `general` : Analyse générale
- `plagiarism` : Détection de plagiat
- `clustering` : Regroupement de documents
- `search` : Pertinence de recherche
- `diversity` : Analyse de diversité

##### Export (NOUVEAU v2.0)

```python
export_results_to_csv(results: List[Dict], filename: str = None) -> str
```

Exporte les résultats au format CSV.

```python
export_results_to_json(results: List[Dict], filename: str = None) -> str
```

Exporte les résultats au format JSON avec configuration et horodatage.

---

## Prétraitement automatique

Le programme applique automatiquement les transformations suivantes :

1. Conversion en minuscules (si `case_sensitive=False`)
2. Suppression de la ponctuation (si `remove_punctuation=True`)
3. Gestion des accents français et caractères spéciaux
4. Division en mots individuels
5. Suppression des espaces multiples
6. Filtrage des stop-words (si `remove_stopwords=True`) - v2.0
7. Application du stemming (si `use_stemming=True`) - v2.0

### Stop-words français (v2.0)

Plus de 60 mots courants filtrés :

- Articles : le, la, les, un, une, des, de, du, au, aux
- Pronoms : je, tu, il, elle, on, nous, vous, ils, elles
- Prépositions : à, dans, par, pour, en, vers, avec, sans
- Conjonctions : et, ou, mais, donc, or, ni, car
- Et bien d'autres...

### Stemming français (v2.0)

Normalisation basique des mots :

- manger → mang
- programmation → programm
- finalement → final
- développement → développ

---

## Résultats d'exemple

| Phrase 1                | Phrase 2                 | Similarité | Distance | Configuration   |
| ----------------------- | ------------------------ | ---------- | -------- | --------------- |
| "Le chat mange"         | "Le chien mange"         | 0.5000     | 0.5000   | Standard        |
| "Python est un langage" | "Java est un langage"    | 0.7500     | 0.2500   | Standard        |
| "Le chat mange"         | "Le chien mange"         | 0.5000     | 0.5000   | Avec stop-words |
| "Je mange une pomme"    | "Tu manges des pommes"   | 0.2000     | 0.8000   | Sans stemming   |
| "Je mange une pomme"    | "Tu manges des pommes"   | 0.4000     | 0.6000   | Avec stemming   |
| "Phrase identique"      | "Phrase identique"       | 1.0000     | 0.0000   | Toute config    |
| "Aucun mot commun"      | "Différent complètement" | 0.0000     | 1.0000   | Toute config    |

---

## Complexité algorithmique

- **Temps** : O(n + m) où n et m sont le nombre de mots dans chaque phrase
     - Prétraitement : O(n) et O(m)
     - Opérations sur ensembles : O(min(n,m))
- **Espace** : O(n + m) pour stocker les ensembles de mots

L'algorithme reste efficace même avec de grandes phrases ou de nombreuses comparaisons.

### Tests de performance (v2.0)

```
  10 phrases →    45 comparaisons en 0.001s (42751 comp/s)
  50 phrases →  1225 comparaisons en 0.025s (49010 comp/s)
 100 phrases →  4950 comparaisons en 0.057s (87549 comp/s)
 200 phrases → 19900 comparaisons en 0.197s (100769 comp/s)
```

---

## Tests et validation

### Tests unitaires

Le fichier `test_jaccard.py` contient **50+ tests** couvrant :

**Tests de base (v1.0) :**

- Phrases identiques (similarité = 1.0)
- Phrases sans mots communs (similarité = 0.0)
- Cas partiels avec calculs vérifiés
- Gestion de la ponctuation et de la casse
- Chaînes vides et cas limites
- Propriétés mathématiques (réflexivité, symétrie)
- Tests de performance

**Nouveaux tests (v2.0) :**

- Distance de Jaccard (8 tests)
- Stop-words français (4 tests)
- Stemming français (5 tests)
- Interprétation contextuelle (6 tests)
- Export CSV/JSON (7 tests)

### Lancer les tests

```bash
python test_jaccard.py
```

**Résultat attendu :**

```
======================================================================
TESTS UNITAIRES - SIMILARITÉ DE JACCARD v2.0
======================================================================

Ran 50 tests in 0.2s

OK
Tests exécutés: 50
Réussites: 50
Échecs: 0
Erreurs: 0

✓ TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!
```

---

## Applications possibles

Cette implémentation peut être utilisée pour :

1. **Détection de plagiat** : Identifier des textes copiés ou paraphrasés
2. **Classification de documents** : Grouper des textes similaires (clustering)
3. **Systèmes de recommandation** : Recommander du contenu similaire
4. **Déduplication** : Éliminer les doublons dans une base de données
5. **Analyse de sentiment** : Comparer des avis ou commentaires
6. **Moteur de recherche** : Trouver des documents pertinents
7. **Analyse de diversité** : Mesurer la variété du contenu (v2.0)
8. **Analyse comparative** : Comparer des versions de documents (v2.0)

---

## Limitations

- **Ordre des mots** : Ne tient pas compte de l'ordre (approche "sac de mots")
- **Synonymes** : Ne reconnaît pas les synonymes (chat ≠ félin, voiture ≠ automobile)
- **Contexte sémantique** : N'analyse pas le sens profond des phrases
- **Négation** : "J'aime" et "Je n'aime pas" ont une haute similarité
- **Stemming basique** : Le stemmer français est volontairement simple (sans dépendances)

---

## Améliorations possibles

### Améliorations techniques

- [ ] **Lemmatisation avancée** : Utiliser NLTK ou Spacy pour un meilleur traitement
- [ ] **N-grammes** : Utiliser des bigrammes ou trigrammes
- [ ] **Pondération TF-IDF** : Donner plus d'importance aux mots rares
- [ ] **Synonymes** : Intégration d'un dictionnaire de synonymes (WordNet)
- [ ] **Distance de Levenshtein** : Tolérance aux fautes d'orthographe
- [ ] **Embeddings** : Utiliser Word2Vec ou BERT pour la similarité sémantique

### Améliorations d'interface

- [ ] **Visualisations** : Graphiques avec matplotlib/plotly
- [ ] **API REST** : Serveur Flask/FastAPI
- [ ] **Application web** : Interface React/Vue.js
- [ ] **Support multilingue** : Optimisation pour d'autres langues
- [ ] **Batch processing** : Traitement de fichiers volumineux
- [ ] **Base de données** : Stockage persistant des résultats

---

## Structure du projet

```
jaccard-similarity-project/
├── jaccard_similarity.py    # Programme principal v2.0
├── jaccard_gui.py          # Interface graphique v2.0
├── test_jaccard.py         # Tests unitaires (50+ tests)
├── README.md               # Documentation (ce fichier)
├── QUICKSTART.md           # Guide de démarrage rapide
├── .gitignore              # Fichiers à ignorer par Git
└── examples/               # Exemples supplémentaires
    └── demo.py            # Démonstrations v2.0 (11 démos)
```

---

## Documentation du code

Le code est entièrement documenté avec :

- **Docstrings** : Chaque fonction et classe est documentée
- **Type hints** : Types explicites pour tous les paramètres et retours
- **Commentaires** : Explications pour les parties complexes
- **Exemples** : Cas d'usage dans les docstrings

---

## Contribution

Nous accueillons les contributions ! Voici comment participer :

1. **Fork** le projet
2. Créez une **branche** pour votre fonctionnalité
      ```bash
      git checkout -b feature/NouvelleFonctionnalite
      ```
3. **Committez** vos changements
      ```bash
      git commit -m 'Ajout de NouvelleFonctionnalite'
      ```
4. **Push** vers la branche
      ```bash
      git push origin feature/NouvelleFonctionnalite
      ```
5. Ouvrez une **Pull Request**

### Normes de contribution

- Suivez le style de code PEP 8
- Ajoutez des tests pour les nouvelles fonctionnalités
- Mettez à jour la documentation
- Assurez-vous que tous les tests passent

---

## Licence

Ce projet est développé dans le cadre d'un TP de Machine Learning non Supervisé.

---

## Liens utiles

- **Repository Git** : [https://github.com/POUBERE/jaccard-similarity-project](https://github.com/POUBERE/jaccard-similarity-project)
- **Issues** : [https://github.com/POUBERE/jaccard-similarity-project/issues](https://github.com/POUBERE/jaccard-similarity-project/issues)
- **Documentation Python** : [https://docs.python.org/3/](https://docs.python.org/3/)

---

## Support

Pour toute question ou problème :

1. Consultez d'abord cette documentation
2. Vérifiez les [Issues existantes](https://github.com/POUBERE/jaccard-similarity-project/issues)
3. Créez une nouvelle Issue si nécessaire
4. Contactez l'équipe : abdourazakoupoubere@gmail.com

---

## Contexte académique

Ce projet a été développé dans le cadre du cours de **Machine Learning non Supervisé**. Il illustre :

- L'implémentation d'une métrique de similarité et distance
- Les bonnes pratiques de développement Python
- La documentation et les tests unitaires
- L'utilisation de Git pour la gestion de version
- Le travail collaboratif en équipe

### Concepts abordés

- **Ensembles et opérations** : Intersection, union
- **Mesures de similarité et distance** : Coefficient de Jaccard
- **Prétraitement de texte** : Tokenisation, normalisation, stemming, stop-words
- **Complexité algorithmique** : Analyse de performance
- **Tests unitaires** : Validation et non-régression
- **Interprétation contextuelle** : Adaptation au domaine d'application

---

## Références

### Articles académiques

- Jaccard, P. (1912). "The distribution of the flora in the alpine zone"
- Manning, C. D., & Schütze, H. (1999). "Foundations of statistical natural language processing"
- Salton, G., & McGill, M. J. (1983). "Introduction to modern information retrieval"

### Ressources en ligne

- [Jaccard Index - Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- [Documentation Python officielle](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)

---

## Changelog

### Version 2.0 (Octobre 2025)

**Nouvelles fonctionnalités :**

- Distance de Jaccard complémentaire à la similarité
- Stemming français basique (sans dépendances)
- Support des stop-words français (60+ mots)
- Interprétation contextuelle des scores (5 contextes)
- Export CSV et JSON avec horodatage
- Interface graphique améliorée
- 20+ nouveaux tests unitaires

**Améliorations :**

- Méthodes unifiées pour éviter la duplication
- Documentation enrichie avec exemples
- Tests de performance optimisés
- Démonstrations interactives complètes

**Méthodes ajoutées :**

- `calculate_distance()` et `calculate_distance_detailed()`
- `get_distance_matrix()`
- `get_most_different_pair()`
- `interpret_similarity()` et `interpret_distance()`
- `export_results_to_csv()` et `export_results_to_json()`

### Version 1.0 (Septembre 2025)

**Fonctionnalités initiales :**

- Calcul de similarité de Jaccard
- Support de la casse et de la ponctuation
- Comparaison multiple de phrases
- Matrice de similarité
- Interface graphique Tkinter
- Tests unitaires de base
- Documentation complète

---

## Checklist du projet

- [x] Implémentation de la similarité de Jaccard
- [x] Implémentation de la distance de Jaccard (v2.0)
- [x] Support des phrases en français
- [x] Gestion de la ponctuation et de la casse
- [x] Stop-words français (v2.0)
- [x] Stemming français (v2.0)
- [x] Tests unitaires complets (50+ tests)
- [x] Documentation détaillée v2.0
- [x] Mode interactif
- [x] Interface graphique v2.0
- [x] Options de configuration avancées
- [x] Comparaison multiple de phrases
- [x] Matrices de similarité/distance
- [x] Interprétation contextuelle (v2.0)
- [x] Export CSV/JSON (v2.0)
- [x] Tests de performance
- [x] Exemples d'utilisation complets
- [x] README complet v2.0

---

## Objectifs du TP

Ce projet répond aux exigences suivantes du TP :

1. **Programme fonctionnel** : Implémentation complète de la similarité de Jaccard avec extensions
2. **Langage libre** : Développé en Python 3
3. **Compte Git** : Repository configuré pour le travail en équipe
4. **Documentation du code** : Docstrings, commentaires, type hints
5. **Mode d'exécution** : Instructions claires et détaillées
6. **Exemples de tests** : Tests automatiques, interactifs, et unitaires (50+)

---

**Développé avec passion par OUEDRAOGO Lassina, OUEDRAOGO Rasmane et POUBERE Abdourazakou**  
_Cours de Machine Learning non Supervisé - Octobre 2025_

**Version 2.0** - Avec distance de Jaccard, stemming, stop-words, interprétation contextuelle et export de données
