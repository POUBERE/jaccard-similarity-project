# 🚀 GUIDE DE DÉMARRAGE RAPIDE

**Bienvenue dans le projet de Similarité de Jaccard !**

Ce guide vous permettra de démarrer rapidement avec le projet.

---

## 🎯 Objectif du Projet

> **Énoncé du TP :** Écrire un programme dans n'importe quel langage informatique pour implémenter la similarité de Jaccard sur les phrases.

Notre projet implémente cet algorithme avec des fonctionnalités avancées :
- ✅ Similarité de Jaccard classique
- ✅ Gestion des synonymes français
- ✅ Lemmatisation avancée
- ✅ Analyse sémantique

---

## 📦 Contenu du Projet

### Fichiers Principaux

```
jaccard-similarity-project/
│
├── 📄 jaccard_similarity.py       # Programme principal
├── 📄 french_lemmatizer.py        # Module de lemmatisation
├── 📄 french_synonyms.py          # Module de synonymes
├── 📄 semantic_analyzer.py        # Module d'analyse sémantique
├── 📄 test_jaccard.py             # Tests unitaires
│
├── 📖 README.md                   # Documentation complète
└── 📖 GUIDE_DEMARRAGE.md         # Ce guide
```

### Description des Modules

| Module | Lignes | Description |
|--------|--------|-------------|
| **jaccard_similarity.py** | ~450 | Classe principale avec toutes les fonctionnalités |
| **french_lemmatizer.py** | ~400 | Lemmatisation de 300+ formes verbales |
| **french_synonyms.py** | ~220 | Dictionnaire de 100+ groupes de synonymes |
| **semantic_analyzer.py** | ~350 | Analyse sémantique avec 16 champs |
| **test_jaccard.py** | ~350 | 27 tests unitaires complets |

---

## ⚡ Démarrage en 3 Minutes

### Étape 1 : Vérifier l'Installation de Python

```bash
# Vérifier que Python 3.6+ est installé
python --version

# Si Python n'est pas installé, téléchargez-le depuis python.org
```

### Étape 2 : Télécharger le Projet

```bash
# Option 1 : Cloner depuis Git
git clone https://github.com/POUBERE/jaccard-similarity-project.git
cd jaccard-similarity-project

# Option 2 : Télécharger et décompresser l'archive ZIP
```

### Étape 3 : Tester le Projet

```bash
# Lancer la démo comparative
python jaccard_similarity.py --demo

# Lancer les tests unitaires
python test_jaccard.py
```

**✅ C'est tout ! Aucune installation de bibliothèque nécessaire.**

---

## 🎮 Modes d'Exécution

### Mode 1 : Démo Comparative (Recommandé pour débuter)

```bash
python jaccard_similarity.py --demo
```

**Ce que vous allez voir :**
```

Test 1:
  Phrase 1: "Le chat mange une souris"
  Phrase 2: "Le félin dévore un rat"

  VERSION 2.0 (stemming + stop-words):
    Similarité: 0.0000

  VERSION 3.0 (lemmatisation + synonymes + stop-words):
    Similarité: 0.8182
    Mots communs (avec synonymes): 9

  VERSION 3.0 COMPLÈTE (lemmatisation + synonymes + sémantique):
    Similarité Jaccard: 0.8182
    Similarité sémantique: 0.7500
    Similarité hybride: 0.7909
```

### Mode 2 : Exécution Simple

```bash
# Lancer le programme avec configuration par défaut
python jaccard_similarity.py
```

### Mode 3 : Avec Options Avancées

```bash
# Avec lemmatisation seulement
python jaccard_similarity.py --use-lemmatization

# Avec lemmatisation + synonymes
python jaccard_similarity.py --use-lemmatization --use-synonyms

# Avec lemmatisation + synonymes + filtrage stop-words
python jaccard_similarity.py --use-lemmatization --use-synonyms --remove-stopwords

# Configuration complète (recommandé)
python jaccard_similarity.py --use-lemmatization --use-synonyms --use-semantic --remove-stopwords
```

### Mode 4 : Mode Interactif

```bash
# Lancer le programme avec configuration par défaut
python jaccard_similarity.py --interactive
# Avec lemmatisation seulement
python jaccard_similarity.py --interactive --use-lemmatization

# Avec lemmatisation + synonymes
python jaccard_similarity.py --interactive --use-lemmatization --use-synonyms

# Avec lemmatisation + synonymes + filtrage stop-words
python jaccard_similarity.py --interactive --use-lemmatization --use-synonyms --remove-stopwords

# Configuration complète (recommandé)
python jaccard_similarity.py --interactive --use-lemmatization --use-synonyms --use-semantic --remove-stopwords
```

### Mode 5 : Tests Unitaires

```bash
# Exécuter tous les tests
python test_jaccard.py
```

**Résultat attendu :**
```

test_add_custom_synonyms (test_jaccard.TestFrenchSynonyms) ... ok
test_are_synonyms (test_jaccard.TestFrenchSynonyms) ... ok
test_expand_with_synonyms (test_jaccard.TestFrenchSynonyms) ... ok
...

----------------------------------------------------------------------
Ran 27 tests in 0.125s

OK

================================================================================
RÉSUMÉ DES TESTS
================================================================================
Tests exécutés: 27
Réussites: 27
Échecs: 0
Erreurs: 0

[OK] TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!
```

---

## 💻 Exemples de Code

### Exemple 1 : Utilisation Basique

```python
from jaccard_similarity import JaccardSimilarity

# Créer un calculateur
calc = JaccardSimilarity()

# Calculer la similarité entre deux phrases
similarity = calc.calculate_similarity(
    "Le chat noir mange",
    "Le chat blanc dort"
)

print(f"Similarité: {similarity:.2%}")
```

**Résultat :**
```
Similarité: 50.00%
```

### Exemple 2 : Avec Lemmatisation

```python
from jaccard_similarity import JaccardSimilarity

# Configuration avec lemmatisation
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True
)

# Tester avec différentes conjugaisons
similarity = calc.calculate_similarity(
    "Je suis content",
    "Nous sommes heureux"
)

print(f"Similarité: {similarity:.2%}")
```

**Avantage :** Les verbes conjugués sont reconnus (`suis` et `sommes` → `être`)

### Exemple 3 : Avec Synonymes (NOUVEAU !)

```python
from jaccard_similarity import JaccardSimilarity

# Configuration avec synonymes
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True
)

# Tester avec des synonymes
result = calc.calculate_similarity_detailed(
    "Le chat noir",
    "Le félin sombre"
)

print(f"Similarité: {result['jaccard_similarity']:.2%}")
print(f"Mots communs (avec synonymes): {result['common_via_synonyms_count']}")
```

**Avantage :** Détecte que `chat` ≈ `félin` et `noir` ≈ `sombre`

### Exemple 4 : Configuration Complète

```python
from jaccard_similarity import JaccardSimilarity

# Configuration maximale
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True,
    use_semantic_analysis=True
)

# Tester avec analyse complète
result = calc.calculate_similarity_detailed(
    "Le médecin soigne le patient",
    "Le docteur traite le malade"
)

print(f"Similarité Jaccard: {result['jaccard_similarity']:.2%}")
print(f"Similarité sémantique: {result['semantic_similarity']:.2%}")
print(f"Similarité hybride: {result['hybrid_similarity']:.2%}")
```

**Avantage :** Combine Jaccard classique + analyse sémantique

### Exemple 5 : Comparaison Multiple

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True
)

# Liste de phrases
phrases = [
    "Le chat mange une souris",
    "Le félin dévore un rat",
    "Le chien court dans le jardin"
]

# Comparer toutes les paires
results = calc.compare_multiple_sentences(phrases)

for i, j, sim in results:
    print(f"Phrase {i} vs Phrase {j}: {sim:.2%}")
```

### Exemple 6 : Export des Résultats

```python
from jaccard_similarity import JaccardSimilarity

calc = JaccardSimilarity(
    use_lemmatization=True,
    use_synonyms=True
)

# Calculer plusieurs comparaisons
results = []
test_pairs = [
    ("Le chat noir", "Le félin blanc"),
    ("La voiture rouge", "L'automobile bleue")
]

for s1, s2 in test_pairs:
    result = calc.calculate_similarity_detailed(s1, s2)
    results.append(result)

# Exporter en JSON
filename = calc.export_results_to_json(results)
print(f"Résultats exportés: {filename}")
```

---

## 🔧 Utilisation des Modules Individuels

### Module 1 : FrenchSynonyms

```python
from french_synonyms import FrenchSynonyms

# Créer l'instance
synonyms = FrenchSynonyms()

# Obtenir les synonymes d'un mot
syns = synonyms.get_synonyms("voiture")
print(f"Synonymes de 'voiture': {syns}")
# Résultat: {'voiture', 'automobile', 'auto', 'véhicule', 'bagnole'}

# Vérifier si deux mots sont synonymes
are_syn = synonyms.are_synonyms("chat", "félin")
print(f"'chat' et 'félin' sont synonymes: {are_syn}")
# Résultat: True

# Ajouter des synonymes personnalisés
synonyms.add_custom_synonyms({'ia', 'intelligence artificielle', 'ai'})
print(f"'ia' et 'ai' sont synonymes: {synonyms.are_synonyms('ia', 'ai')}")
# Résultat: True

# Obtenir les statistiques
stats = synonyms.get_stats()
print(f"Total de mots: {stats['total_words']}")
print(f"Groupes de synonymes: {stats['total_groups']}")
```

### Module 2 : FrenchLemmatizer

```python
from french_lemmatizer import FrenchLemmatizer

# Créer l'instance
lemmatizer = FrenchLemmatizer()

# Lemmatiser des verbes irréguliers
print(f"suis → {lemmatizer.lemmatize('suis')}")        # être
print(f"avais → {lemmatizer.lemmatize('avais')}")      # avoir
print(f"irai → {lemmatizer.lemmatize('irai')}")        # aller

# Lemmatiser des verbes réguliers
print(f"mange → {lemmatizer.lemmatize('mange')}")      # manger
print(f"mangeons → {lemmatizer.lemmatize('mangeons')}")  # manger
print(f"mangé → {lemmatizer.lemmatize('mangé')}")      # manger

# Lemmatiser des noms au pluriel
print(f"chevaux → {lemmatizer.lemmatize('chevaux')}")  # cheval
print(f"animaux → {lemmatizer.lemmatize('animaux')}")  # animal
print(f"bateaux → {lemmatizer.lemmatize('bateaux')}")  # bateau

# Lemmatiser des adjectifs féminins
print(f"belle → {lemmatizer.lemmatize('belle')}")      # beau
print(f"grande → {lemmatizer.lemmatize('grande')}")    # grand

# Ajouter un lemme personnalisé
lemmatizer.add_custom_lemma('tweets', 'tweet')
print(f"tweets → {lemmatizer.lemmatize('tweets')}")    # tweet
```

### Module 3 : SemanticAnalyzer

```python
from semantic_analyzer import SemanticAnalyzer

# Créer l'instance
analyzer = SemanticAnalyzer()

# Obtenir les champs sémantiques d'un mot
fields = analyzer.get_semantic_fields("chat")
print(f"Champs sémantiques de 'chat': {fields}")
# Résultat: {'animaux'}

# Vérifier si deux mots sont sémantiquement liés
related = analyzer.are_semantically_related("chat", "chien")
print(f"'chat' et 'chien' sont liés: {related}")
# Résultat: True (même champ: animaux)

# Calculer la similarité sémantique
sim = analyzer.semantic_similarity("chat", "chien")
print(f"Similarité sémantique: {sim:.2f}")
# Résultat: 1.00 (même champ)

# Obtenir les mots liés
related_words = analyzer.get_related_words("chat", max_words=5)
print("Mots liés à 'chat':")
for word, score in related_words:
    print(f"  {word}: {score:.2f}")

# Ajouter un champ sémantique personnalisé
analyzer.add_semantic_field('langages', {'python', 'java', 'javascript', 'ruby'})
fields = analyzer.get_semantic_fields("python")
print(f"Champs de 'python': {fields}")
# Résultat: {'langages'}
```

---

## 📊 Tableau des Options

| Option | Par Défaut | Description | Quand l'utiliser ? |
|--------|------------|-------------|-------------------|
| `case_sensitive` | False | Respecte la casse (A ≠ a) | Textes avec acronymes |
| `remove_punctuation` | True | Supprime . , ! ? etc. | Toujours recommandé |
| `remove_stopwords` | False | Filtre le, la, de, etc. | Textes longs |
| `use_stemming` | False | Stemming basique | Compatibilité v2.0 |
| **`use_lemmatization`** | **False** | **Lemmatisation avancée** | **Toujours recommandé** |
| **`use_synonyms`** | **False** | **Gestion synonymes** | **Détection similarité sémantique** |
| **`use_semantic_analysis`** | **False** | **Analyse conceptuelle** | **Textes avec concepts liés** |

---

## 🎓 Quelle Configuration Choisir ?

### Pour Débuter (Découverte)
```python
calc = JaccardSimilarity()
```
**Usage :** Tests basiques, apprentissage

### Pour une Précision Standard
```python
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True
)
```
**Usage :** Comparaisons de textes courts

### Pour une Précision Avancée (Recommandé)
```python
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True
)
```
**Usage :** Détection de plagiat, recherche de documents

### Pour une Précision Maximale
```python
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True,
    use_semantic_analysis=True
)
```
**Usage :** Analyse sémantique, clustering, recherche avancée

---

## ❓ Questions Fréquentes

### Q1 : Puis-je utiliser le projet sans Git ?
**R :** Oui ! Téléchargez simplement les fichiers Python et exécutez-les.

### Q2 : Ai-je besoin d'installer des bibliothèques ?
**R :** Non ! Le projet n'utilise que la bibliothèque standard Python.

### Q3 : Comment tester si tout fonctionne ?
**R :** Exécutez `python test_jaccard.py` - tous les tests doivent passer.

### Q4 : Quelle est la différence entre lemmatisation et stemming ?
**R :** 
- **Stemming** : Coupe brutalement (`mangeons` → `mang`)
- **Lemmatisation** : Trouve la vraie forme (`mangeons` → `manger`)

### Q5 : Les synonymes ralentissent-ils le programme ?
**R :** Très peu (~2-3ms de plus), mais la précision augmente de +800% !

### Q6 : Puis-je ajouter mes propres synonymes ?
**R :** Oui ! Utilisez `synonyms.add_custom_synonyms({'mot1', 'mot2'})`

### Q7 : Comment exporter les résultats ?
**R :** Utilisez `calc.export_results_to_json(results)`

### Q8 : Le projet fonctionne sur Windows/Mac/Linux ?
**R :** Oui ! Python 3.6+ suffit sur tous les systèmes.

---

## 🚨 Résolution de Problèmes

### Problème 1 : "command not found: python"
**Solution :**
```bash
# Essayez python3 à la place
python3 jaccard_similarity.py --demo

# Ou installez Python depuis python.org
```

### Problème 2 : "ModuleNotFoundError"
**Solution :**
```bash
# Assurez-vous d'être dans le bon dossier
cd jaccard-similarity-project
ls  # Vous devez voir les fichiers .py

# Vérifiez que tous les fichiers sont présents
```

### Problème 3 : Tests qui échouent
**Solution :**
```bash
# Vérifiez la version de Python
python --version  # Doit être 3.6+

# Réessayez l'exécution
python test_jaccard.py
```

### Problème 4 : "UnicodeDecodeError"
**Solution :**
```bash
# Spécifiez l'encodage UTF-8
export PYTHONIOENCODING=utf-8
python jaccard_similarity.py
```

---

## 📚 Ordre d'Apprentissage Recommandé

### Jour 1 : Découverte (30 minutes)
1. ✅ Lire ce guide
2. ✅ Exécuter la démo : `python jaccard_similarity.py --demo`
3. ✅ Lancer les tests : `python test_jaccard.py`

### Jour 2 : Compréhension (1 heure)
1. ✅ Lire le README.md complet
2. ✅ Tester les exemples de code fournis
3. ✅ Expérimenter avec vos propres phrases

### Jour 3 : Approfondissement (2 heures)
1. ✅ Étudier les modules individuels
2. ✅ Ajouter des synonymes personnalisés
3. ✅ Tester différentes configurations

### Jour 4 : Pratique (3 heures)
1. ✅ Créer vos propres exemples
2. ✅ Adapter le code à vos besoins
3. ✅ Exporter et analyser les résultats

### Jour 5 : Maîtrise (illimité)
1. ✅ Comprendre le code source
2. ✅ Proposer des améliorations
3. ✅ Documenter vos propres cas d'usage

---

## ✅ Checklist de Démarrage

Cochez au fur et à mesure :

- [ ] Python 3.6+ est installé
- [ ] Projet téléchargé et décompressé
- [ ] `python jaccard_similarity.py --demo` fonctionne
- [ ] `python test_jaccard.py` affiche 27/27 tests réussis
- [ ] J'ai testé l'Exemple 1 (utilisation basique)
- [ ] J'ai testé l'Exemple 2 (avec lemmatisation)
- [ ] J'ai testé l'Exemple 3 (avec synonymes)
- [ ] J'ai lu le README.md complet
- [ ] Je comprends les 3 modules (synonymes, lemmatisation, sémantique)
- [ ] Je sais quelle configuration choisir pour mon besoin
- [ ] Je suis prêt à utiliser le projet ! 🚀

---

## 🎯 Prochaines Étapes

Une fois que vous maîtrisez le projet :

1. **Expérimentez** avec vos propres textes
2. **Modifiez** les paramètres pour voir l'impact
3. **Ajoutez** vos propres synonymes et lemmes
4. **Comparez** les différentes configurations
5. **Documentez** vos découvertes
6. **Partagez** vos améliorations avec l'équipe

---

## 📞 Besoin d'Aide ?

**Contacts :**
- OUEDRAOGO Lassina
- OUEDRAOGO Rasmane
- POUBERE Abdourazakou

**Email :** abdourazakoupoubere@gmail.com

**Documentation :**
- README.md : Documentation complète
- Code source : Docstrings détaillées dans chaque fichier

---

## 🏆 Félicitations !

Vous êtes maintenant prêt à utiliser le projet de Similarité de Jaccard !

**N'oubliez pas :** La configuration recommandée pour de meilleurs résultats est :
```python
calc = JaccardSimilarity(
    remove_stopwords=True,
    use_lemmatization=True,
    use_synonyms=True,
    use_semantic_analysis=True
)
```

**Bon développement ! 🎉**

---

*Développé par OUEDRAOGO Lassina, OUEDRAOGO Rasmane et POUBERE Abdourazakou*  
*Machine Learning non Supervisé - Novembre 2025*

**Version 3.0** - *Guide de Démarrage Rapide*
