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

## 🧮 Complexité algorithmique

- **Temps** : O(n + m) où n et m sont le nombre de mots dans chaque phrase
     - Prétraitement : O(n) et O(m)
     - Opérations sur ensembles (intersection, union) : O(min(n,m))
- **Espace** : O(n + m) pour stocker les ensembles de mots

L'algorithme reste efficace même avec de grandes phrases ou de nombreuses comparaisons.

## 🔍 Tests et validation

### Tests unitaires inclus

Le fichier `test_jaccard.py` contient plus de 30 tests couvrant :

- ✅ Phrases identiques (similarité = 1.0)
- ✅ Phrases sans mots communs (similarité = 0.0)
- ✅ Cas partiels avec calculs vérifiés
- ✅ Gestion de la ponctuation
- ✅ Sensibilité à la casse
- ✅ Chaînes vides
- ✅ Propriétés mathématiques (réflexivité, symétrie)
- ✅ Cas limites (espaces, caractères spéciaux)
- ✅ Tests de performance
- ✅ Exemples du monde réel

### Lancer les tests

```bash
python test_jaccard.py
```

Les tests affichent également un résumé des performances avec différentes tailles de données.

## 📈 Applications possibles

Cette implémentation peut être utilisée pour :

- 🔍 **Détection de plagiat** : Identifier des textes copiés ou paraphrasés
- 📚 **Classification de documents** : Grouper des textes similaires (clustering)
- 🤖 **Systèmes de recommandation** : Recommander du contenu similaire
- 🔗 **Déduplication** : Éliminer les doublons dans une base de données
- 📊 **Analyse de sentiment** : Comparer des avis ou commentaires
- 🔎 **Moteur de recherche** : Trouver des documents pertinents par rapport à une requête
- 📝 **Analyse de texte** : Étudier la similarité entre corpus de textes

## ⚠️ Limitations

- **Ordre des mots** : Ne tient pas compte de l'ordre (approche "sac de mots")
- **Synonymes** : Ne reconnaît pas les synonymes (chat ≠ félin, voiture ≠ automobile)
- **Contexte sémantique** : N'analyse pas le sens profond des phrases
- **Négation** : "J'aime" et "Je n'aime pas" ont une haute similarité
- **Longueur** : Sensible aux différences de longueur entre phrases

## 🚀 Améliorations possibles

### Améliorations techniques

- [ ] **Stemming/Lemmatisation** : Réduire les mots à leur racine
- [ ] **N-grammes** : Utiliser des bigrammes ou trigrammes au lieu de mots uniques
- [ ] **Pondération TF-IDF** : Donner plus d'importance aux mots rares
- [ ] **Stop words** : Liste personnalisée de mots à ignorer
- [ ] **Synonymes** : Intégration d'un dictionnaire de synonymes
- [ ] **Distance de Levenshtein** : Tolérance aux fautes d'orthographe

### Améliorations d'interface

- [ ] **Interface graphique** : GUI avec Tkinter ou PyQt
- [ ] **API REST** : Serveur Flask/FastAPI pour utilisation web
- [ ] **Visualisations** : Graphiques de similarité avec matplotlib
- [ ] **Export de résultats** : CSV, JSON, Excel
- [ ] **Support multilingue** : Optimisation pour différentes langues
- [ ] **Batch processing** : Traitement de fichiers volumineux

## 📁 Structure du projet

```
jaccard-similarity-project/
├── jaccard_similarity.py    # Programme principal
├── test_jaccard.py         # Tests unitaires complets
├── README.md               # Documentation (ce fichier)
├── .gitignore              # Fichiers à ignorer par Git
├── LICENSE                 # Licence du projet
└── examples/               # Exemples supplémentaires
    └── demo.py            # Script de démonstration avancée
```

## 📚 Documentation du code

Le code est entièrement documenté avec :

- **Docstrings** : Chaque fonction et classe est documentée
- **Type hints** : Types explicites pour tous les paramètres et retours
- **Commentaires** : Explications pour les parties complexes
- **Exemples** : Cas d'usage dans les docstrings

### Exemple de documentation

```python
def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
    """
    Calcule la similarité de Jaccard entre deux phrases.

    Args:
        sentence1: Première phrase
        sentence2: Deuxième phrase

    Returns:
        Similarité de Jaccard (entre 0 et 1)

    Exemple:
        >>> calculator = JaccardSimilarity()
        >>> calculator.calculate_similarity("Le chat mange", "Le chien mange")
        0.5
    """
```

## 🐛 Résolution de problèmes

### Problème : ImportError

**Erreur** : `ModuleNotFoundError: No module named 'jaccard_similarity'`

**Solution** : Assurez-vous d'être dans le bon répertoire et que le fichier `jaccard_similarity.py` existe.

```bash
# Vérification du répertoire
ls -la
# Doit afficher jaccard_similarity.py

# Exécution depuis le bon répertoire
python jaccard_similarity.py
```

### Problème : Encodage de caractères

**Erreur** : Problèmes avec les accents français

**Solution** : Le fichier utilise l'encodage UTF-8. Vérifiez la configuration de votre terminal :

```bash
# Sur Linux/Mac
export LANG=fr_FR.UTF-8

# Sur Windows (PowerShell)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### Problème : Performances lentes

**Symptôme** : Calculs trop longs avec beaucoup de phrases

**Solution** : Pour de grandes quantités de données, optimisez :

```python
# Évitez les comparaisons redondantes
# Utilisez get_similarity_matrix() au lieu de multiples calculate_similarity()

# Pour n phrases, utilisez :
matrix = calculator.get_similarity_matrix(sentences)
# Au lieu de n² appels individuels
```

## 🤝 Contribution

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

## 📄 Licence

Ce projet est développé dans le cadre d'un TP de Machine Learning non Supervisé.

## 🔗 Liens utiles

- **Repository Git** : [https://github.com/[votre-username]/jaccard-similarity-project](https://github.com/POUBERE/jaccard-similarity-project)
- **Issues** : [https://github.com/[votre-username]/jaccard-similarity-project/issues](https://github.com/POUBERE/jaccard-similarity-project/issues)
- **Documentation Python** : [https://docs.python.org/3/](https://docs.python.org/3/)

## 📞 Support

Pour toute question ou problème :

1. Consultez d'abord cette documentation
2. Vérifiez les [Issues existantes](https://github.com/POUBERE/jaccard-similarity-project/issues)
3. Créez une nouvelle Issue si nécessaire
4. Contactez l'équipe : [abdourazakoupoubere@gmail.com]

## 🎓 Contexte académique

Ce projet a été développé dans le cadre du cours de **Machine Learning non Supervisé**. Il illustre :

- L'implémentation d'une métrique de similarité
- Les bonnes pratiques de développement Python
- La documentation et les tests unitaires
- L'utilisation de Git pour la gestion de version
- Le travail collaboratif en équipe

### Concepts abordés

- **Ensembles et opérations** : Intersection, union
- **Mesures de similarité** : Coefficient de Jaccard
- **Prétraitement de texte** : Tokenisation, normalisation
- **Complexité algorithmique** : Analyse de performance
- **Tests unitaires** : Validation et non-régression

## 📖 Références

### Articles académiques

- Jaccard, P. (1912). "The distribution of the flora in the alpine zone"
- Manning, C. D., & Schütze, H. (1999). "Foundations of statistical natural language processing"

### Ressources en ligne

- [Introduction à la similarité de Jaccard](https://en.wikipedia.org/wiki/Jaccard_index)
- [Documentation Python officielle](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)

## ✅ Checklist du projet

- [x] Implémentation de la similarité de Jaccard
- [x] Support des phrases en français
- [x] Gestion de la ponctuation et de la casse
- [x] Tests unitaires complets (30+ tests)
- [x] Documentation détaillée
- [x] Mode interactif
- [x] Options de configuration
- [x] Comparaison multiple de phrases
- [x] Matrice de similarité
- [x] Tests de performance
- [x] Exemples d'utilisation
- [x] README complet

## 🎯 Objectifs du TP

Ce projet répond aux exigences suivantes du TP :

1. ✅ **Programme fonctionnel** : Implémentation complète de la similarité de Jaccard
2. ✅ **Langage libre** : Développé en Python 3
3. ✅ **Compte Git** : Repository configuré pour le travail en équipe
4. ✅ **Documentation du code** : Docstrings, commentaires, type hints
5. ✅ **Mode d'exécution** : Instructions claires et détaillées
6. ✅ **Exemples de tests** : Tests automatiques et interactifs

---

**Développé avec ❤️ par OUEDRAOGO Lassina, OUEDRAOGO Rasmane et POUBERE Abdourazakou**  
_Cours de Machine Learning non Supervisé - Septembre 2025_
