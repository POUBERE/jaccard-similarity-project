J'ai fait des amélioration Dans la version 2.0 je pouvais lancer le mdoe interactive afin de saisir les mot ou phrase moi mais la version 3.0 ne fait pas sa. tu peux intégrer sa pour moi ? :

Voici les commandes pour le mode interactif :
### 2. Mode interactif

```bash
python jaccard_similarity.py --interactive

# Prise en compte de la casse
python jaccard_similarity.py --interactive --case-sensitive

# Suppression des stop-words français
python jaccard_similarity.py --interactive --remove-stopwords

# Utilisation du stemming français
python jaccard_similarity.py --interactive --use-stemming

# Combinaison d'options
python jaccard_similarity.py --interactive --remove-stopwords --use-stemming

```

Voici l'énoncé du projet :
Projet de Machine Learning non Supervisé
TP (Pour le prochain cours) : Écrire un programme dans n'importe quel langage informatique pour implémenter la similarité Jaccard sur les phrases.
Créer un compte Git pour votre groupe.
Travailler sur la documentation de votre code.
La documentation doit inclure le mode d'exécution de votre programme et des exemples de tests.

Voici la version 3.0 du projet :

# jaccard_similarity.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de calcul de similarité de Jaccard entre phrases
Projet de Machine Learning non Supervisé - VERSION v2.0

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

La similarité de Jaccard mesure la ressemblance entre deux ensembles
en calculant le rapport entre l'intersection et l'union des ensembles.
Formule: Jaccard(A,B) = |A ∩ B| / |A ∪ B|
"""

# ============================================================================
# IMPORTS
# ============================================================================
import re  # Pour le nettoyage de texte avec les regex
import argparse  # Pour gérer les arguments en ligne de commande
import json  # Pour l'export JSON
import csv  # Pour l'export CSV
from typing import Set, List, Tuple, Dict  # Pour les annotations de type
from datetime import datetime  # Pour les timestamps dans les exports


# ============================================================================
# CLASSE FrenchStemmer
# Cette classe réduit les mots français à leur racine
# Exemple: "manger", "mange", "mangé" deviennent tous "mang"
# ============================================================================
class FrenchStemmer:
    """Stemmer pour le français avec gestion des cas spéciaux."""

    # Liste des suffixes français, triés du plus long au plus court
    # Important: l'ordre évite de couper trop tôt (ex: "ation" avant "s")
    SUFFIXES = [
        'issements', 'issement',
        'atrice', 'ations', 'ation', 'atrices',
        'erions', 'eraient', 'assent', 'assiez', 'èrent',
        'erons', 'eront', 'erait', 'eriez', 'erais',
        'ements', 'ement', 'euses', 'euse', 'istes', 'iste',
        'ables', 'able', 'ances', 'ance', 'ences', 'ence',
        'ments', 'ment', 'ités', 'ité', 'eurs', 'eur',
        'eaux', 'aux', 'ant', 'ent', 'ait', 'ais',
        'er', 'es', 'é', 'ée', 'és', 'ées', 's'
    ]

    # Mots qu'on ne doit jamais stemmer
    # Ce sont des mots grammaticaux qui perdraient leur sens
    PROTECTED_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'de', 'du', 'au', 'aux', 'ce', 'ces',
        'et', 'ou', 'mais', 'car', 'or', 'donc', 'ni',
        'si', 'ne', 'pas', 'plus', 'très', 'bien', 'tout'
    }

    # Dictionnaire des verbes irréguliers
    # On associe les formes conjuguées à une racine commune
    EXCEPTIONS = {
        'suis': 'êtr', 'es': 'êtr', 'est': 'êtr',
        'sommes': 'êtr', 'êtes': 'êtr', 'sont': 'êtr',
        'ai': 'av', 'as': 'av', 'a': 'av',
        'avons': 'av', 'avez': 'av', 'ont': 'av',
        'vais': 'all', 'va': 'all', 'allons': 'all',
        'allez': 'all', 'vont': 'all',
    }

    @staticmethod
    def stem(word: str) -> str:
        """
        Applique le stemming à un mot français.

        Paramètres:
            word (str): Le mot à traiter

        Retourne:
            str: La racine du mot

        Logique:
            On suit plusieurs étapes pour décider comment traiter le mot
        """
        # Les mots trop courts (≤2 caractères) sont laissés tels quels
        if len(word) <= 2:
            return word.lower()

        word_lower = word.lower()

        # Vérifier si c'est un mot protégé
        if word_lower in FrenchStemmer.PROTECTED_WORDS:
            return word_lower

        # Vérifier si c'est une forme irrégulière connue
        if word_lower in FrenchStemmer.EXCEPTIONS:
            return FrenchStemmer.EXCEPTIONS[word_lower]

        # Essayer d'enlever un suffixe
        for suffix in FrenchStemmer.SUFFIXES:
            if word_lower.endswith(suffix):
                stem_candidate = word_lower[:-len(suffix)]
                # On garde la racine seulement si elle fait au moins 3 caractères
                if len(stem_candidate) >= 3:
                    return stem_candidate

        # Si aucune règle ne marche, on retourne le mot en minuscules
        return word_lower


# ============================================================================
# FONCTIONS DE VALIDATION
# Ces fonctions vérifient que les données sont correctes avant traitement
# ============================================================================

def validate_sentence(sentence: str, allow_empty: bool = False) -> bool:
    """
    Valide une phrase avant de la traiter.

    Paramètres:
        sentence (str): La phrase à valider
        allow_empty (bool): Autorise les phrases vides si True

    Retourne:
        bool: True si tout est bon

    Lève une exception si:
        - Ce n'est pas une chaîne de caractères
        - La phrase est vide (sauf si allow_empty=True)
        - La phrase est trop longue
    """
    # Vérifier le type
    if not isinstance(sentence, str):
        raise TypeError(
            f"La phrase doit être une chaîne de caractères, pas {type(sentence).__name__}")

    # Vérifier si vide
    if not allow_empty and not sentence.strip():
        raise ValueError("La phrase ne peut pas être vide")

    # Vérifier la longueur (limite à 10000 caractères)
    if len(sentence) > 10000:
        raise ValueError(
            f"La phrase est trop longue ({len(sentence)} caractères, max 10000)")

    return True


def validate_sentences_list(sentences: List[str], min_length: int = 2) -> bool:
    """
    Valide une liste de phrases.

    Paramètres:
        sentences (List[str]): Liste à valider
        min_length (int): Nombre minimum de phrases requis

    Retourne:
        bool: True si tout est bon

    Cette fonction est utilisée avant les opérations sur plusieurs phrases
    (comme les matrices ou les comparaisons multiples)
    """
    # Vérifier que c'est bien une liste
    if not isinstance(sentences, list):
        raise TypeError(
            f"sentences doit être une liste, pas {type(sentences).__name__}")

    # Vérifier qu'il y a assez de phrases
    if len(sentences) < min_length:
        raise ValueError(
            f"Au moins {min_length} phrases sont requises, {len(sentences)} fournies")

    # Vérifier chaque phrase individuellement
    for i, sentence in enumerate(sentences):
        try:
            validate_sentence(sentence)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Phrase {i} invalide: {e}")

    return True


# ============================================================================
# CLASSE JaccardSimilarity
# C'est la classe principale qui fait tous les calculs
# ============================================================================
class JaccardSimilarity:
    """Classe pour calculer la similarité de Jaccard entre phrases."""

    # Liste des stop-words français (mots très fréquents qui apportent peu de sens)
    FRENCH_STOPWORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
        'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
        'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'lui', 'y', 'en',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
        'à', 'dans', 'par', 'pour', 'en', 'vers', 'avec', 'sans', 'sous', 'sur',
        'qui', 'que', 'quoi', 'dont', 'où',
        'si', 'ne', 'pas', 'plus', 'moins', 'très', 'tout', 'toute', 'tous', 'toutes',
        'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
        'falloir', 'vouloir', 'venir', 'devoir', 'croire', 'trouver', 'donner',
        'prendre', 'parler', 'aimer', 'passer', 'mettre'
    }

    def __init__(self, case_sensitive: bool = False, remove_punctuation: bool = True,
                 remove_stopwords: bool = False, use_stemming: bool = False):
        """
        Initialise le calculateur avec les options choisies.

        Paramètres:
            case_sensitive: Si True, "Python" et "python" sont différents
            remove_punctuation: Si True, enlève la ponctuation
            remove_stopwords: Si True, filtre les stop-words
            use_stemming: Si True, applique le stemming
        """
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

        # On crée le stemmer seulement si on en a besoin
        self.stemmer = FrenchStemmer() if use_stemming else None

    def preprocess_sentence(self, sentence: str) -> Set[str]:
        """
        Prétraite une phrase et la convertit en ensemble de mots.

        Paramètre:
            sentence (str): La phrase à traiter

        Retourne:
            Set[str]: Ensemble des mots (sans doublons)

        Étapes du traitement:
            1. Normalisation de la casse
            2. Suppression de la ponctuation
            3. Découpage en mots
            4. Filtrage des stop-words
            5. Stemming

        On utilise un Set parce que Jaccard travaille sur des ensembles,
        donc les répétitions ne comptent pas
        """
        # Normalisation de la casse
        if not self.case_sensitive:
            sentence = sentence.lower()

        # Suppression de la ponctuation avec une regex
        # On garde seulement les lettres et les espaces
        if self.remove_punctuation:
            sentence = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', sentence)

        # Découpage en mots
        words = [word.strip() for word in sentence.split() if word.strip()]

        # Filtrage des stop-words si activé
        if self.remove_stopwords:
            words = [w for w in words if w.lower() not in self.FRENCH_STOPWORDS]

        # Application du stemming si activé
        if self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(w) for w in words]

        # Conversion en Set
        return set(words)

    def calculate_similarity_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la similarité de Jaccard avec tous les détails.

        Paramètres:
            sentence1, sentence2: Les deux phrases à comparer

        Retourne:
            Dict: Dictionnaire avec:
                - Les phrases originales
                - Les ensembles de mots
                - L'intersection et l'union
                - Le score de similarité

        Formule: Similarité = |A ∩ B| / |A ∪ B|
        """
        # Validation
        validate_sentence(sentence1, allow_empty=True)
        validate_sentence(sentence2, allow_empty=True)

        # Prétraitement
        set1 = self.preprocess_sentence(sentence1)
        set2 = self.preprocess_sentence(sentence2)

        # Calcul de l'intersection (mots communs)
        intersection = set1.intersection(set2)

        # Calcul de l'union (tous les mots uniques)
        union = set1.union(set2)

        # Calcul de la similarité
        # Si l'union est vide, on retourne 0
        similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'words_set1': set1,
            'words_set2': set2,
            'intersection': intersection,
            'union': union,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'jaccard_similarity': similarity
        }

    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Version simple qui retourne juste le score de similarité.

        Retourne un float entre 0 et 1:
            - 0 = aucun mot commun
            - 1 = phrases identiques
        """
        validate_sentence(sentence1, allow_empty=True)
        validate_sentence(sentence2, allow_empty=True)

        result = self.calculate_similarity_detailed(sentence1, sentence2)
        return result['jaccard_similarity']

    def calculate_distance_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la distance de Jaccard avec tous les détails.

        La distance est le complément de la similarité: Distance = 1 - Similarité

        Propriétés de la distance:
            - Distance entre phrases identiques = 0
            - Distance entre phrases sans mots communs = 1
            - La distance respecte l'inégalité triangulaire
        """
        detailed = self.calculate_similarity_detailed(sentence1, sentence2)
        detailed['jaccard_distance'] = 1.0 - detailed['jaccard_similarity']
        return detailed

    def calculate_distance(self, sentence1: str, sentence2: str) -> float:
        """
        Version simple qui retourne juste le score de distance.
        """
        result = self.calculate_distance_detailed(sentence1, sentence2)
        return result['jaccard_distance']

    def compare_multiple_sentences(self, sentences: List[str]) -> List[Tuple[int, int, float]]:
        """
        Compare toutes les paires possibles dans une liste de phrases.

        Paramètre:
            sentences: Liste des phrases

        Retourne:
            Liste de tuples (index1, index2, similarité)

        Pour n phrases, on fait n(n-1)/2 comparaisons
        Exemple: 4 phrases → 6 comparaisons

        Complexité: O(n²)
        """
        validate_sentences_list(sentences, min_length=2)

        results = []

        # Double boucle pour générer toutes les paires
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self.calculate_similarity(
                    sentences[i], sentences[j])
                results.append((i, j, similarity))

        return results

    def get_metric_matrix(self, sentences: List[str], metric: str = 'similarity') -> List[List[float]]:
        """
        Calcule une matrice de similarité ou de distance.

        Paramètres:
            sentences: Liste des phrases
            metric: 'similarity' ou 'distance'

        Retourne:
            Matrice n×n (liste de listes)

        La matrice est symétrique, et la diagonale contient:
            - 1.0 pour la similarité (une phrase est identique à elle-même)
            - 0.0 pour la distance

        Utile pour les algorithmes de clustering, les visualisations, etc.
        """
        validate_sentences_list(sentences, min_length=1)

        n = len(sentences)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        # Choix de la fonction selon la métrique
        calc_func = self.calculate_similarity if metric == 'similarity' else self.calculate_distance
        diagonal_value = 1.0 if metric == 'similarity' else 0.0

        # Remplissage de la matrice
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = diagonal_value
                else:
                    matrix[i][j] = calc_func(sentences[i], sentences[j])

        return matrix

    def get_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """Raccourci pour calculer la matrice de similarité."""
        return self.get_metric_matrix(sentences, 'similarity')

    def get_distance_matrix(self, sentences: List[str]) -> List[List[float]]:
        """Raccourci pour calculer la matrice de distance."""
        return self.get_metric_matrix(sentences, 'distance')

    def get_extreme_pair(self, sentences: List[str], mode: str = 'most_similar') -> Tuple[int, int, float]:
        """
        Trouve la paire de phrases la plus similaire ou la plus différente.

        Paramètres:
            sentences: Liste des phrases
            mode: 'most_similar' ou 'most_different'

        Retourne:
            Tuple (index1, index2, score)

        Cas d'usage:
            - Détection de plagiat (most_similar)
            - Analyse de diversité (most_different)
        """
        validate_sentences_list(sentences, min_length=2)

        if mode == 'most_similar':
            comparisons = self.compare_multiple_sentences(sentences)
            if not comparisons:
                return (0, 0, 0.0)
            return max(comparisons, key=lambda x: x[2])
        else:
            # Pour la distance max, on fait une recherche manuelle
            max_distance = -1
            max_pair = (0, 0)

            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    distance = self.calculate_distance(
                        sentences[i], sentences[j])
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = (i, j)

            return (max_pair[0], max_pair[1], max_distance)

    def get_most_similar_pair(self, sentences: List[str]) -> Tuple[int, int, float]:
        """Trouve la paire la plus similaire."""
        return self.get_extreme_pair(sentences, 'most_similar')

    def get_most_different_pair(self, sentences: List[str]) -> Tuple[int, int, float]:
        """Trouve la paire la plus différente."""
        return self.get_extreme_pair(sentences, 'most_different')

    def interpret_metric(self, value: float, metric: str = 'similarity', context: str = 'general') -> Dict[str, str]:
        """
        Interprète un score en fonction du contexte.

        Paramètres:
            value: Le score à interpréter
            metric: 'similarity' ou 'distance'
            context: Contexte d'utilisation (plagiarism, clustering, search, diversity, general)

        Retourne:
            Dict avec:
                - score: La valeur numérique
                - category: Catégorie ("Très similaire", etc.)
                - emoji: Émoji visuel
                - color_code: Code couleur
                - general_interpretation: Explication générale
                - contextual_interpretation: Explication selon le contexte
                - recommendations: Recommandations
                - technical_explanation: Explication mathématique

        Catégories:
            1.0 → Identique
            ≥0.8 → Très similaire
            ≥0.6 → Assez similaire
            ≥0.4 → Moyennement similaire
            ≥0.2 → Peu similaire
            >0 → Très peu similaire
            0 → Aucune similarité
        """
        # Conversion entre distance et similarité
        if metric == 'distance':
            similarity = 1.0 - value
            distance = value
        else:
            similarity = value
            distance = 1.0 - value

        # Catégorisation du score
        if similarity == 1.0:
            category = "Identique"
            emoji = "✅"
            color_code = "green"
        elif similarity >= 0.8:
            category = "Très similaire"
            emoji = "🟢"
            color_code = "green"
        elif similarity >= 0.6:
            category = "Assez similaire"
            emoji = "🟡"
            color_code = "yellow"
        elif similarity >= 0.4:
            category = "Moyennement similaire" if metric == 'similarity' else "Moyennement différent"
            emoji = "🟠"
            color_code = "orange"
        elif similarity >= 0.2:
            category = "Peu similaire" if metric == 'similarity' else "Très différent"
            emoji = "🔴"
            color_code = "red"
        elif similarity > 0:
            category = "Très peu similaire"
            emoji = "⚫"
            color_code = "dark_red"
        else:
            category = "Aucune similarité" if metric == 'similarity' else "Complètement différent"
            emoji = "❌"
            color_code = "black"

        # Construction du résultat
        result = {
            'score': value,
            'category': category,
            'emoji': emoji,
            'color_code': color_code,
            'general_interpretation': self._get_unified_general_interpretation(similarity),
            'contextual_interpretation': self._get_unified_contextual_interpretation(similarity, context, metric),
            'recommendations': self._get_unified_recommendations(similarity, context, metric),
            'technical_explanation': self._get_unified_technical_explanation(similarity, distance, metric)
        }

        if metric == 'distance':
            result['similarity'] = similarity
            result['distance'] = distance

        return result

    def interpret_similarity(self, similarity: float, context: str = "general") -> Dict[str, str]:
        """Raccourci pour interpréter une similarité."""
        return self.interpret_metric(similarity, 'similarity', context)

    def interpret_distance(self, distance: float, context: str = "general") -> Dict[str, str]:
        """Raccourci pour interpréter une distance."""
        return self.interpret_metric(distance, 'distance', context)

    def _get_unified_general_interpretation(self, similarity: float) -> str:
        """
        Fournit une explication générale du score.

        Traduit le score numérique en texte compréhensible
        """
        if similarity == 1.0:
            return ("Les deux phrases sont parfaitement identiques. Tous les mots sont communs "
                    "et aucun mot unique n'existe dans l'une ou l'autre phrase.")
        elif similarity >= 0.8:
            return ("Les phrases partagent la grande majorité de leurs mots. Elles expriment "
                    "probablement des idées très proches avec une formulation similaire.")
        elif similarity >= 0.6:
            return ("Les phrases ont une base commune importante mais contiennent aussi des "
                    "différences notables. Elles traitent probablement du même sujet mais "
                    "avec des nuances.")
        elif similarity >= 0.4:
            return ("Les phrases partagent certains mots-clés mais diffèrent sensiblement. "
                    "Elles peuvent traiter de sujets connexes ou utiliser un vocabulaire commun "
                    "dans des contextes différents.")
        elif similarity >= 0.2:
            return ("Les phrases ont quelques mots en commun, probablement des mots fréquents "
                    "ou génériques. Elles sont globalement différentes dans leur contenu.")
        elif similarity > 0:
            return ("Les phrases partagent très peu de mots. Il peut s'agir de mots très "
                    "courants (articles, prépositions) sans lien sémantique fort.")
        else:
            return ("Aucun mot n'est partagé entre les deux phrases. Elles traitent de "
                    "sujets complètement différents ou utilisent des vocabulaires distincts.")

    def _get_unified_contextual_interpretation(self, similarity: float, context: str, metric: str) -> str:
        """
        Fournit une interprétation adaptée au contexte.

        Chaque contexte a ses propres seuils:
            - Plagiarism: score élevé = alerte
            - Clustering: score moyen = même groupe
            - Search: score élevé = pertinent
            - Diversity: score faible = varié
        """
        # Dictionnaire d'interprétations par contexte
        interpretations = {
            'plagiarism': {
                1.0: "🚨 PLAGIAT CERTAIN - Copie intégrale détectée",
                0.8: "⚠️  PLAGIAT TRÈS PROBABLE - Similarité suspecte, nécessite une vérification",
                0.6: "⚠️  SUSPICION ÉLEVÉE - Peut indiquer une paraphrase ou réarrangement",
                0.4: "⚡ SUSPICION MODÉRÉE - Quelques éléments communs, à examiner",
                0.2: "✓ SUSPICION FAIBLE - Probablement du contenu original",
                0.0: "✓ CONTENU ORIGINAL - Aucune similarité détectée"
            },
            'clustering': {
                1.0: "📂 CLUSTER IDENTIQUE - Documents identiques ou doublons",
                0.8: "📂 CLUSTER FORT - Documents très liés, même catégorie",
                0.6: "📂 CLUSTER MODÉRÉ - Documents connexes, possiblement même thème",
                0.4: "📂 CLUSTER FAIBLE - Quelques liens, catégories voisines possibles",
                0.2: "📂 PAS DE CLUSTER - Documents distincts",
                0.0: "📂 TOTALEMENT DISTINCTS - Aucun lien apparent"
            },
            'search': {
                1.0: "🎯 PERTINENCE MAXIMALE - Correspondance parfaite avec la requête",
                0.8: "🎯 TRÈS PERTINENT - Contient la plupart des termes de recherche",
                0.6: "🎯 PERTINENT - Bon match avec plusieurs termes clés",
                0.4: "🎯 PARTIELLEMENT PERTINENT - Contient quelques termes de recherche",
                0.2: "🎯 PEU PERTINENT - Match faible avec la requête",
                0.0: "🎯 NON PERTINENT - Aucun terme de recherche trouvé"
            },
            'diversity': {
                1.0: "🎨 AUCUNE DIVERSITÉ - Contenu identique" if metric == 'distance' else "🎨 IDENTIQUE",
                0.8: "🎨 FAIBLE DIVERSITÉ - Contenu très homogène" if metric == 'distance' else "🎨 TRÈS SIMILAIRE",
                0.6: "🎨 DIVERSITÉ MODÉRÉE - Mix de similarités" if metric == 'distance' else "🎨 ASSEZ SIMILAIRE",
                0.4: "🎨 BONNE DIVERSITÉ - Contenus variés" if metric == 'distance' else "🎨 MOYENNEMENT SIMILAIRE",
                0.2: "🎨 FORTE DIVERSITÉ - Contenus très différents" if metric == 'distance' else "🎨 PEU SIMILAIRE",
                0.0: "🎨 DIVERSITÉ MAXIMALE - Contenus totalement distincts" if metric == 'distance' else "🎨 AUCUNE SIMILARITÉ"
            },
            'general': {
                1.0: "Les phrases sont identiques",
                0.8: "Très haute similarité - Contenu très proche",
                0.6: "Bonne similarité - Sujet probablement commun",
                0.4: "Similarité modérée - Quelques éléments partagés",
                0.2: "Faible similarité - Peu d'éléments communs",
                0.0: "Aucune similarité détectée"
            }
        }

        context_interp = interpretations.get(
            context, interpretations['general'])

        # Sélection selon le score
        if similarity == 1.0:
            return context_interp[1.0]
        elif similarity >= 0.8:
            return context_interp[0.8]
        elif similarity >= 0.6:
            return context_interp[0.6]
        elif similarity >= 0.4:
            return context_interp[0.4]
        elif similarity >= 0.2:
            return context_interp[0.2]
        else:
            return context_interp[0.0]

    def _get_unified_recommendations(self, similarity: float, context: str, metric: str) -> List[str]:
        """
        Fournit des recommandations selon le score et le contexte.
        """
        recommendations = []

        if context == 'plagiarism':
            if similarity >= 0.8:
                recommendations.extend([
                    "Vérifier manuellement le document source",
                    "Comparer les citations et références",
                    "Utiliser des outils de détection plus avancés",
                    "Contacter l'auteur pour clarification"
                ])
            elif similarity >= 0.5:
                recommendations.extend([
                    "Examiner les passages spécifiques similaires",
                    "Vérifier si une paraphrase est appropriée",
                    "S'assurer que les sources sont citées"
                ])

        elif context == 'clustering':
            if similarity >= 0.6:
                recommendations.extend([
                    "Regrouper ces documents dans le même cluster",
                    "Analyser les thèmes communs pour mieux les catégoriser"
                ])
            elif similarity >= 0.3:
                recommendations.append(
                    "Considérer comme potentiellement liés, vérifier manuellement")
            else:
                recommendations.append("Séparer dans des clusters différents")

        elif context == 'search':
            if similarity >= 0.4:
                recommendations.append(
                    "Document pertinent, à inclure dans les résultats")
            else:
                recommendations.append(
                    "Document peu pertinent, peut être exclu des résultats")

        elif context == 'diversity':
            if metric == 'distance':
                if similarity <= 0.3:
                    recommendations.append(
                        "Bonne diversité détectée - Contenu varié")
                else:
                    recommendations.append(
                        "Diversité faible - Envisager d'ajouter du contenu différent")

        if similarity == 0.0:
            recommendations.append(
                "Aucun mot commun - Vérifier le prétraitement des textes" if metric == 'similarity'
                else "Aucun mot commun - Vocabulaires totalement différents")
        elif similarity < 0.3 and len(recommendations) == 0:
            recommendations.append(
                "Similarité faible - Ces textes traitent probablement de sujets différents")

        return recommendations if recommendations else ["Aucune recommandation spécifique"]

    def _get_unified_technical_explanation(self, similarity: float, distance: float, metric: str) -> str:
        """
        Fournit une explication technique du score.
        """
        if metric == 'similarity':
            percentage = similarity * 100
            explanation = f"Score de Jaccard: {similarity:.4f} ({percentage:.2f}%)\n\n"

            if similarity == 1.0:
                explanation += ("L'intersection des ensembles de mots égale leur union. "
                                "Mathématiquement: |A ∩ B| = |A ∪ B|")
            elif similarity >= 0.5:
                explanation += (f"Environ {percentage:.0f}% des mots de l'union sont partagés. "
                                f"Cela signifie qu'environ {100-percentage:.0f}% des mots sont uniques "
                                f"à l'une ou l'autre phrase.")
            else:
                explanation += (f"Seulement {percentage:.0f}% des mots de l'union sont communs. "
                                f"La majorité ({100-percentage:.0f}%) des mots sont spécifiques "
                                f"à chaque phrase.")
        else:
            percentage_diff = distance * 100
            explanation = f"Distance de Jaccard: {distance:.4f} ({percentage_diff:.2f}%)\n"
            explanation += f"Similarité correspondante: {similarity:.4f}\n\n"

            if distance == 0.0:
                explanation += "Distance nulle → Ensembles identiques\n"
                explanation += "Mathématiquement: d(A,B) = 1 - |A ∩ B|/|A ∪ B| = 0"
            elif distance == 1.0:
                explanation += "Distance maximale → Ensembles disjoints\n"
                explanation += "Mathématiquement: |A ∩ B| = 0"
            else:
                explanation += (f"Environ {percentage_diff:.0f}% de dissimilarité entre les ensembles.\n"
                                f"Cela signifie {100-percentage_diff:.0f}% de mots sont partagés.")

        return explanation

    def export_results_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        Exporte les résultats au format CSV.

        Paramètres:
            results: Liste des résultats à exporter
            filename: Nom du fichier (auto-généré si None)

        Retourne:
            str: Nom du fichier créé (ou None si erreur)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jaccard_results_{timestamp}.csv"

        if not results:
            print("Aucun résultat à exporter.")
            return None

        fieldnames = ['sentence1', 'sentence2',
                      'similarity', 'distance', 'category']

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow({
                        'sentence1': result.get('sentence1', ''),
                        'sentence2': result.get('sentence2', ''),
                        'similarity': result.get('jaccard_similarity', 0.0),
                        'distance': result.get('jaccard_distance', 1.0),
                        'category': result.get('category', 'N/A')
                    })

            print(f"✓ Résultats exportés vers: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Erreur lors de l'export CSV: {e}")
            return None

    def export_results_to_json(self, results: List[Dict], filename: str = None) -> str:
        """
        Exporte les résultats au format JSON.

        Paramètres:
            results: Liste des résultats
            filename: Nom du fichier (auto-généré si None)

        Retourne:
            str: Nom du fichier créé (ou None si erreur)

        Le fichier JSON contient:
            - timestamp: Date et heure de l'export
            - config: Configuration utilisée
            - results: Les résultats proprement dits
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jaccard_results_{timestamp}.json"

        if not results:
            print("Aucun résultat à exporter.")
            return None

        try:
            # Conversion des Sets en Lists (JSON ne supporte pas les sets)
            export_data = []
            for result in results:
                export_item = result.copy()

                if 'words_set1' in export_item:
                    export_item['words_set1'] = list(export_item['words_set1'])
                if 'words_set2' in export_item:
                    export_item['words_set2'] = list(export_item['words_set2'])
                if 'intersection' in export_item:
                    export_item['intersection'] = list(
                        export_item['intersection'])
                if 'union' in export_item:
                    export_item['union'] = list(export_item['union'])

                export_data.append(export_item)

            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'case_sensitive': self.case_sensitive,
                        'remove_punctuation': self.remove_punctuation,
                        'remove_stopwords': self.remove_stopwords,
                        'use_stemming': self.use_stemming
                    },
                    'results': export_data
                }, jsonfile, indent=2, ensure_ascii=False)

            print(f"✓ Résultats exportés vers: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Erreur lors de l'export JSON: {e}")
            return None


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def run_example_tests(calculator: JaccardSimilarity, export_format: str = None):
    """
    Exécute des tests avec des exemples prédéfinis.

    Cette fonction montre le fonctionnement du programme avec 5 exemples
    qui couvrent différents cas de figure
    """
    print("=== Programme de Calcul de Similarité de Jaccard ===\n")

    print("Configuration active:")
    print(
        f"  - Sensibilité à la casse: {'Activée' if calculator.case_sensitive else 'Désactivée'}")
    print(
        f"  - Stop-words: {'Activés' if calculator.remove_stopwords else 'Désactivés'}")
    print(
        f"  - Stemming: {'Activé' if calculator.use_stemming else 'Désactivé'}")
    print()

    examples = [
        ("Le chat mange des croquettes", "Le chien mange des croquettes"),
        ("Python est un langage de programmation",
         "Java est un langage de programmation"),
        ("Machine learning supervisé", "Apprentissage automatique supervisé"),
        ("Bonjour tout le monde", "Salut tout le monde"),
        ("Aucun mot en commun", "Différentes phrases complètement")
    ]

    print("1. Tests de base avec double analyse (Similarité + Distance) :")
    print("-" * 80)

    all_results = []

    for i, (s1, s2) in enumerate(examples, 1):
        similarity = calculator.calculate_similarity(s1, s2)
        distance = calculator.calculate_distance(s1, s2)

        sim_interpretation = calculator.interpret_similarity(
            similarity, context='general')
        dist_interpretation = calculator.interpret_distance(
            distance, context='general')

        detailed = calculator.calculate_distance_detailed(s1, s2)
        detailed['category'] = sim_interpretation['category']
        all_results.append(detailed)

        print(f"\nTest {i}:")
        print(f"  Phrase 1: '{s1}'")
        print(f"  Phrase 2: '{s2}'")
        print(f"\n  📊 SIMILARITÉ: {similarity:.4f}")
        print(
            f"     Catégorie: {sim_interpretation['emoji']} {sim_interpretation['category']}")
        print(f"     {sim_interpretation['general_interpretation']}")
        print(f"\n  📏 DISTANCE: {distance:.4f}")
        print(
            f"     Catégorie: {dist_interpretation['emoji']} {dist_interpretation['category']}")
        print(f"     {dist_interpretation['general_interpretation']}")
        print(
            f"\n  ✓ Vérification: Similarité + Distance = {similarity + distance:.4f}")
        print("-" * 80)

    # Export si demandé
    if export_format:
        if export_format.lower() == 'csv':
            calculator.export_results_to_csv(all_results)
        elif export_format.lower() == 'json':
            calculator.export_results_to_json(all_results)
        elif export_format.lower() == 'both':
            calculator.export_results_to_csv(all_results)
            calculator.export_results_to_json(all_results)


def interactive_mode(calculator: JaccardSimilarity):
    """
    Mode interactif pour saisir des phrases manuellement.

    L'utilisateur peut saisir ses propres phrases et voir les résultats.
    Taper 'quit' pour sortir.
    """
    print("=== Mode Interactif - Calculateur de Jaccard ===")
    print(f"Configuration: case_sensitive={calculator.case_sensitive}, "
          f"remove_stopwords={calculator.remove_stopwords}, "
          f"use_stemming={calculator.use_stemming}")
    print("Entrez 'quit' pour quitter\n")

    while True:
        sentence1 = input("Phrase 1: ").strip()
        if sentence1.lower() == 'quit':
            break

        sentence2 = input("Phrase 2: ").strip()
        if sentence2.lower() == 'quit':
            break

        try:
            similarity = calculator.calculate_similarity(sentence1, sentence2)
            distance = calculator.calculate_distance(sentence1, sentence2)

            sim_interpretation = calculator.interpret_similarity(
                similarity, context='general')
            dist_interpretation = calculator.interpret_distance(
                distance, context='general')

            print(f"\n{'='*70}")
            print(f"RÉSULTAT DE LA COMPARAISON")
            print(f"{'='*70}")

            set1 = calculator.preprocess_sentence(sentence1)
            set2 = calculator.preprocess_sentence(sentence2)
            intersection = set1.intersection(set2)

            print(
                f"\nMots communs: {sorted(intersection)} ({len(intersection)} mots)")
            print(f"Total mots uniques: {len(set1.union(set2))} mots")

            print(f"\n{'─'*70}")
            print(f"📊 SIMILARITÉ DE JACCARD")
            print(f"{'─'*70}")
            print(f"Score: {similarity:.4f}")
            print(
                f"Catégorie: {sim_interpretation['emoji']} {sim_interpretation['category']}")
            print(f"\n💡 Interprétation:")
            print(f"   {sim_interpretation['general_interpretation']}")

            print(f"\n{'─'*70}")
            print(f"📏 DISTANCE DE JACCARD")
            print(f"{'─'*70}")
            print(f"Score: {distance:.4f}")
            print(
                f"Catégorie: {dist_interpretation['emoji']} {dist_interpretation['category']}")
            print(f"\n💡 Interprétation:")
            print(f"   {dist_interpretation['general_interpretation']}")

            print(
                f"\n✓ Vérification: Similarité ({similarity:.4f}) + Distance ({distance:.4f}) = {similarity + distance:.4f}")
            print("-" * 70)
            print()

        except (ValueError, TypeError) as e:
            print(f"\n❌ Erreur: {e}\n")


def main():
    """
    Fonction principale du programme.

    Gère l'interface en ligne de commande et lance le mode approprié.
    """
    parser = argparse.ArgumentParser(
        description='Calcul de similarité et distance de Jaccard entre phrases',
        epilog='Exemples:\n'
               '  python jaccard_similarity.py\n'
               '  python jaccard_similarity.py --case-sensitive\n'
               '  python jaccard_similarity.py --remove-stopwords --use-stemming\n'
               '  python jaccard_similarity.py --case-sensitive --export both',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--case-sensitive', action='store_true',
                        help='Respecte la casse des mots (Python ≠ python)')
    parser.add_argument('--keep-punctuation', action='store_true',
                        help='Garde la ponctuation')
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='Retire les stop-words français (le, la, les, etc.)')
    parser.add_argument('--use-stemming', action='store_true',
                        help='Applique le stemming français (manger → mang)')
    parser.add_argument('--interactive', action='store_true',
                        help='Mode interactif pour saisir des phrases')
    parser.add_argument('--export', choices=['csv', 'json', 'both'],
                        help='Exporte les résultats (csv, json, ou both)')
    parser.add_argument('--run-tests', action='store_true',
                        help='Exécute les tests unitaires')

    args = parser.parse_args()

    if args.run_tests:
        run_unit_tests()
        return

    calculator = JaccardSimilarity(
        case_sensitive=args.case_sensitive,
        remove_punctuation=not args.keep_punctuation,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming
    )

    if args.interactive:
        interactive_mode(calculator)
    else:
        run_example_tests(calculator, args.export)


def run_unit_tests():
    """
    Placeholder pour les tests unitaires.

    Pour l'instant, cette fonction redirige vers un fichier de tests séparé.
    """
    print("⚠️  Pour exécuter les tests complets, utilisez:")
    print("    python test_jaccard.py")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    main()
```

# test_jaccard.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le calculateur de similarité de Jaccard
Projet de Machine Learning non Supervisé

Usage: python test_jaccard.py
"""

import unittest
import time
import sys
import os

from jaccard_similarity import JaccardSimilarity


class TestJaccardSimilarityBasic(unittest.TestCase):
    """Classe pour tester les fonctionnalités de base."""

    def setUp(self):
        """Initialisation des calculateurs pour les tests."""
        self.calculator = JaccardSimilarity()
        self.calculator_case_sensitive = JaccardSimilarity(case_sensitive=True)
        self.calculator_with_punct = JaccardSimilarity(
            remove_punctuation=False)

    def test_identical_sentences(self):
        """Vérification qu'une phrase identique à elle-même donne 1.0."""
        sentence = "Le chat mange des croquettes"
        similarity = self.calculator.calculate_similarity(sentence, sentence)
        self.assertEqual(similarity, 1.0)

    def test_completely_different_sentences(self):
        """Deux phrases complètement différentes doivent donner 0.0."""
        sentence1 = "Le chat mange"
        sentence2 = "Python programmation"
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        self.assertEqual(similarity, 0.0)

    def test_partial_similarity(self):
        """Test du calcul avec des phrases qui ont des mots en commun."""
        sentence1 = "Le chat mange des croquettes"
        sentence2 = "Le chien mange des croquettes"

        # 4 mots en commun (le, mange, des, croquettes)
        # 6 mots au total (le, chat, chien, mange, des, croquettes)
        # Donc 4/6 = 0.6667
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        self.assertAlmostEqual(similarity, 4/6, places=4)

    def test_empty_sentences(self):
        """Test du comportement avec des chaînes vides."""
        similarity_both_empty = self.calculator.calculate_similarity("", "")
        self.assertEqual(similarity_both_empty, 0.0)

        similarity_one_empty = self.calculator.calculate_similarity(
            "", "hello world")
        self.assertEqual(similarity_one_empty, 0.0)

    def test_single_word_sentences(self):
        """Comparaison de phrases d'un seul mot."""
        similarity_same = self.calculator.calculate_similarity("chat", "chat")
        self.assertEqual(similarity_same, 1.0)

        similarity_diff = self.calculator.calculate_similarity("chat", "chien")
        self.assertEqual(similarity_diff, 0.0)


class TestPreprocessing(unittest.TestCase):
    """Tests du prétraitement des phrases."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_preprocess_basic(self):
        """Test simple du prétraitement."""
        result = self.calculator.preprocess_sentence("Hello World")
        expected = {'hello', 'world'}
        self.assertEqual(result, expected)

    def test_preprocess_punctuation(self):
        """La ponctuation doit être supprimée."""
        result = self.calculator.preprocess_sentence("Hello, World!")
        expected = {'hello', 'world'}
        self.assertEqual(result, expected)

    def test_preprocess_empty(self):
        """Une phrase vide doit retourner un ensemble vide."""
        result = self.calculator.preprocess_sentence("")
        expected = set()
        self.assertEqual(result, expected)

    def test_preprocess_accents(self):
        """Les accents français doivent être préservés."""
        result = self.calculator.preprocess_sentence("Café français")
        expected = {'café', 'français'}
        self.assertEqual(result, expected)

    def test_preprocess_multiple_spaces(self):
        """Les espaces multiples doivent être gérés correctement."""
        result = self.calculator.preprocess_sentence("Le  chat   mange")
        expected = {'le', 'chat', 'mange'}
        self.assertEqual(result, expected)

    def test_preprocess_spaces_only(self):
        """Une phrase avec que des espaces doit donner un ensemble vide."""
        result = self.calculator.preprocess_sentence("   ")
        self.assertEqual(result, set())


class TestCaseAndPunctuation(unittest.TestCase):
    """Tests des options de casse et ponctuation."""

    def test_case_sensitivity_off(self):
        """Par défaut, la casse ne devrait pas être prise en compte."""
        calculator = JaccardSimilarity(case_sensitive=False)
        similarity = calculator.calculate_similarity(
            "Hello World", "hello world")
        self.assertEqual(similarity, 1.0)

    def test_case_sensitivity_on(self):
        """Quand case_sensitive=True, la casse doit être respectée."""
        calculator = JaccardSimilarity(case_sensitive=True)
        sentence1 = "Hello World"
        sentence2 = "hello world"

        similarity = calculator.calculate_similarity(sentence1, sentence2)
        self.assertLess(similarity, 1.0)

    def test_punctuation_removal(self):
        """La ponctuation est supprimée par défaut."""
        calculator = JaccardSimilarity(remove_punctuation=True)
        similarity = calculator.calculate_similarity(
            "Hello, world!", "Hello world")
        self.assertEqual(similarity, 1.0)

    def test_punctuation_kept(self):
        """Avec remove_punctuation=False, la ponctuation est gardée."""
        calculator = JaccardSimilarity(remove_punctuation=False)
        similarity = calculator.calculate_similarity("Hello!", "Hello")
        self.assertLess(similarity, 1.0)


class TestDetailedCalculation(unittest.TestCase):
    """Tests pour le calcul détaillé."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_detailed_result_structure(self):
        """Vérification que le résultat détaillé contient toutes les informations."""
        result = self.calculator.calculate_similarity_detailed(
            "hello world", "hello python"
        )

        # On vérifie que toutes les clés sont présentes
        required_keys = [
            'sentence1', 'sentence2', 'words_set1', 'words_set2',
            'intersection', 'union', 'intersection_size', 'union_size',
            'jaccard_similarity'
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_detailed_calculation_values(self):
        """Test des valeurs retournées par le calcul détaillé."""
        result = self.calculator.calculate_similarity_detailed(
            "hello world", "hello python"
        )

        self.assertEqual(result['words_set1'], {'hello', 'world'})
        self.assertEqual(result['words_set2'], {'hello', 'python'})
        self.assertEqual(result['intersection'], {'hello'})
        self.assertEqual(result['union'], {'hello', 'world', 'python'})
        self.assertEqual(result['intersection_size'], 1)
        self.assertEqual(result['union_size'], 3)
        self.assertAlmostEqual(result['jaccard_similarity'], 1/3, places=3)


class TestMultipleComparisons(unittest.TestCase):
    """Tests pour comparer plusieurs phrases à la fois."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_compare_multiple_sentences(self):
        """Test de la comparaison de plusieurs phrases."""
        sentences = [
            "Le chat mange",
            "Le chien mange",
            "Python programmation"
        ]

        results = self.calculator.compare_multiple_sentences(sentences)

        # Avec 3 phrases, on devrait avoir 3 comparaisons: (0,1), (0,2), (1,2)
        self.assertEqual(len(results), 3)

        # Chaque résultat doit être bien formaté
        for idx1, idx2, similarity in results:
            self.assertIsInstance(idx1, int)
            self.assertIsInstance(idx2, int)
            self.assertIsInstance(similarity, float)
            self.assertTrue(0 <= similarity <= 1)
            self.assertLess(idx1, idx2)

    def test_get_most_similar_pair(self):
        """Recherche de la paire la plus similaire dans une liste."""
        sentences = [
            "Le chat mange des croquettes",
            "Python est génial",
            "Le chien mange des croquettes",
            "Java est bien"
        ]

        idx1, idx2, max_similarity = self.calculator.get_most_similar_pair(
            sentences)

        # Les phrases 0 et 2 devraient être les plus similaires
        self.assertTrue((idx1 == 0 and idx2 == 2) or (idx1 == 2 and idx2 == 0))
        self.assertGreater(max_similarity, 0.5)

    def test_similarity_matrix(self):
        """Test de la génération d'une matrice de similarité."""
        sentences = ["chat", "chien", "oiseau"]
        matrix = self.calculator.get_similarity_matrix(sentences)

        # La matrice doit être 3x3
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)

        # La diagonale doit contenir des 1.0
        for i in range(3):
            self.assertEqual(matrix[i][i], 1.0)

        # La matrice doit être symétrique
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(matrix[i][j], matrix[j][i], places=10)


class TestRealWorldExamples(unittest.TestCase):
    """Tests avec des cas réalistes."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_similar_news_articles(self):
        """Test avec des titres d'actualité similaires."""
        news1 = "Le président annonce de nouvelles mesures économiques"
        news2 = "Le chef de l'État dévoile des mesures pour l'économie"
        similarity = self.calculator.calculate_similarity(news1, news2)
        self.assertGreater(similarity, 0.0)

    def test_programming_languages(self):
        """Test avec des phrases sur la programmation."""
        s1 = "Python est un langage de programmation"
        s2 = "Java est un langage de programmation"
        similarity = self.calculator.calculate_similarity(s1, s2)

        # 5 mots communs (est, un, langage, de, programmation)
        # 7 mots au total
        expected = 5/7
        self.assertAlmostEqual(similarity, expected, places=3)

    def test_animal_sentences(self):
        """Test avec des phrases sur les animaux."""
        s1 = "Le chat mange des croquettes"
        s2 = "Le chien mange des os"
        similarity = self.calculator.calculate_similarity(s1, s2)

        # 3 mots communs (le, mange, des)
        # 7 mots au total
        expected = 3/7
        self.assertAlmostEqual(similarity, expected, places=3)


class TestMathematicalProperties(unittest.TestCase):
    """Vérification des propriétés mathématiques de Jaccard."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_range_property(self):
        """La similarité doit toujours être entre 0 et 1."""
        sentences = [
            "chat mange",
            "chien court",
            "oiseau vole",
            "poisson nage"
        ]

        for s1 in sentences:
            for s2 in sentences:
                similarity = self.calculator.calculate_similarity(s1, s2)
                self.assertTrue(0 <= similarity <= 1,
                                f"Similarité hors limites: {similarity}")

    def test_reflexivity(self):
        """Une phrase comparée à elle-même doit toujours donner 1."""
        sentences = ["chat", "chien court", "oiseau vole rapidement"]

        for sentence in sentences:
            similarity = self.calculator.calculate_similarity(
                sentence, sentence)
            self.assertEqual(similarity, 1.0,
                             f"Réflexivité échouée pour '{sentence}'")

    def test_symmetry(self):
        """Jaccard(A,B) doit être égal à Jaccard(B,A)."""
        pairs = [
            ("chat mange", "chien court"),
            ("python code", "java programmation"),
            ("bonjour monde", "hello world")
        ]

        for s1, s2 in pairs:
            sim1 = self.calculator.calculate_similarity(s1, s2)
            sim2 = self.calculator.calculate_similarity(s2, s1)
            self.assertAlmostEqual(sim1, sim2, places=10,
                                   msg=f"Symétrie échouée pour '{s1}' et '{s2}'")


class TestEdgeCases(unittest.TestCase):
    """Tests de cas particuliers et limites."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_only_punctuation(self):
        """Phrases composées uniquement de ponctuation."""
        similarity = self.calculator.calculate_similarity("!!!", "???")
        self.assertEqual(similarity, 0.0)

    def test_repeated_words(self):
        """Les mots répétés ne comptent qu'une fois dans les ensembles."""
        s1 = "chat chat chat"
        s2 = "chat"
        similarity = self.calculator.calculate_similarity(s1, s2)
        self.assertEqual(similarity, 1.0)

    def test_very_long_sentences(self):
        """Test avec des phrases très longues pour vérifier la robustesse."""
        long_sentence1 = " ".join(["mot"] * 100)
        long_sentence2 = " ".join(["mot"] * 50)
        similarity = self.calculator.calculate_similarity(
            long_sentence1, long_sentence2)
        self.assertEqual(similarity, 1.0)

    def test_special_characters(self):
        """Test avec des caractères spéciaux."""
        s1 = "hello@world.com"
        s2 = "hello world com"
        similarity = self.calculator.calculate_similarity(s1, s2)
        self.assertGreater(similarity, 0.0)
        self.assertAlmostEqual(similarity, 1.0, places=2)


class TestPerformance(unittest.TestCase):
    """Tests de performance du calculateur."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_large_sentences_performance(self):
        """Mesure du temps de calcul avec de grandes phrases."""
        words = [f"mot{i}" for i in range(1000)]
        sentence1 = " ".join(words[:800])
        sentence2 = " ".join(words[200:])

        start_time = time.time()
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        end_time = time.time()

        # Ça devrait prendre moins d'une seconde
        self.assertLess(end_time - start_time, 1.0)

        # On vérifie aussi que le résultat est cohérent
        self.assertTrue(0 <= similarity <= 1)

    def test_many_comparisons_performance(self):
        """Test de performance avec beaucoup de comparaisons."""
        sentences = [f"phrase numéro {i} avec des mots" for i in range(50)]

        start_time = time.time()
        results = self.calculator.compare_multiple_sentences(sentences)
        end_time = time.time()

        # On doit avoir 50*49/2 = 1225 comparaisons
        expected_comparisons = 50 * 49 // 2
        self.assertEqual(len(results), expected_comparisons)

        # Ça devrait prendre moins de 2 secondes
        self.assertLess(end_time - start_time, 2.0)


def run_performance_summary():
    """Affichage d'un résumé des performances."""
    print("\n" + "="*70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*70)

    calculator = JaccardSimilarity()

    test_sizes = [10, 50, 100, 200]

    for size in test_sizes:
        sentences = [
            f"phrase de test {i} avec quelques mots" for i in range(size)]

        start_time = time.time()
        results = calculator.compare_multiple_sentences(sentences)
        end_time = time.time()

        execution_time = end_time - start_time
        comparisons = len(results)
        comp_per_sec = comparisons / \
            execution_time if execution_time > 0 else float('inf')

        print(f"  {size:3d} phrases → {comparisons:5d} comparaisons en {execution_time:.3f}s "
              f"({comp_per_sec:.0f} comp/s)")


if __name__ == "__main__":
    print("="*70)
    print("TESTS UNITAIRES - SIMILARITÉ DE JACCARD")
    print("="*70)

    unittest.main(verbosity=2, exit=False)

    run_performance_summary()

    print("\n" + "="*70)
    print("TESTS TERMINÉS")
    print("="*70)
```

# jaccard_gui.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique COMPLÈTE pour le calculateur de similarité de Jaccard
Version 2.2 - TOUTES LES FONCTIONNALITÉS

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

Implémente TOUTES les fonctionnalités de jaccard_similarity.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict
import sys
import os
from datetime import datetime

from jaccard_similarity import JaccardSimilarity, FrenchStemmer


class JaccardGUI:
    """Interface graphique complète avec TOUTES les fonctionnalités."""

    def __init__(self, root):
        """Initialise l'interface graphique."""
        self.root = root
        self.root.title("Calculateur de Jaccard v2.2 - Interface Complète")

        self.root.geometry("1300x950")
        self.root.minsize(1000, 750)

        # Configuration responsive
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=0)
        self.root.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'light': '#ECF0F1',
            'dark': '#34495E',
            'purple': '#9B59B6'
        }

        self.calculator = JaccardSimilarity()

        # Variables pour les options
        self.case_sensitive = tk.BooleanVar(value=False)
        self.remove_punctuation = tk.BooleanVar(value=True)
        self.remove_stopwords = tk.BooleanVar(value=False)
        self.use_stemming = tk.BooleanVar(value=False)
        self.context_var = tk.StringVar(value='general')

        self.phrases_list = []
        self.create_widgets()

    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        self.create_header()
        self.create_options_frame()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)

        # Tous les onglets avec TOUTES les fonctionnalités
        self.create_simple_comparison_tab()
        self.create_multiple_comparison_tab()
        self.create_matrix_tab()
        self.create_extreme_pairs_tab()  # NOUVEAU
        self.create_demo_tests_tab()  # NOUVEAU - Tests automatiques
        self.create_export_tab()
        self.create_about_tab()

        self.create_status_bar()

    def create_header(self):
        """Crée l'en-tête."""
        header_frame = tk.Frame(
            self.root, bg=self.colors['primary'], height=80)
        header_frame.grid(row=0, column=0, sticky='ew')
        header_frame.grid_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📊 Calculateur de Jaccard v2.2 - Interface Complète",
            font=('Arial', 20, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="TOUTES les fonctionnalités de jaccard_similarity.py",
            font=('Arial', 10),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        subtitle_label.pack()

    def create_options_frame(self):
        """Crée le cadre des options."""
        options_frame = tk.LabelFrame(
            self.root,
            text="⚙️ Configuration Complète",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        options_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        row1 = tk.Frame(options_frame)
        row1.pack(fill='x', pady=2)

        tk.Checkbutton(
            row1, text="Sensible à la casse",
            variable=self.case_sensitive,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Supprimer ponctuation",
            variable=self.remove_punctuation,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Retirer stop-words",
            variable=self.remove_stopwords,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Stemming français",
            variable=self.use_stemming,
            command=self.update_calculator
        ).pack(side='left', padx=15)

    def update_calculator(self):
        """Met à jour le calculateur."""
        self.calculator = JaccardSimilarity(
            case_sensitive=self.case_sensitive.get(),
            remove_punctuation=self.remove_punctuation.get(),
            remove_stopwords=self.remove_stopwords.get(),
            use_stemming=self.use_stemming.get()
        )
        self.update_status("Configuration mise à jour")

    def create_simple_comparison_tab(self):
        """Onglet de comparaison simple COMPLET."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Simple  ")

        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        scrollbar.grid(row=0, column=1, sticky='ns')

        scrollable_frame.columnconfigure(0, weight=1)

        # Phrase 1
        tk.Label(scrollable_frame, text="Phrase 1:", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase1_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase1_text.grid(row=1, column=0, sticky='ew', pady=5, padx=10)

        # Phrase 2
        tk.Label(scrollable_frame, text="Phrase 2:", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase2_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase2_text.grid(row=3, column=0, sticky='ew', pady=5, padx=10)

        # Contexte
        context_frame = tk.LabelFrame(
            scrollable_frame,
            text="🎯 Contexte d'interprétation",
            font=('Arial', 10, 'bold'),
            padx=10, pady=5
        )
        context_frame.grid(row=4, column=0, sticky='ew', pady=10, padx=10)

        contexts = [
            ('Général', 'general'),
            ('Plagiat', 'plagiarism'),
            ('Clustering', 'clustering'),
            ('Recherche', 'search'),
            ('Diversité', 'diversity')
        ]

        context_inner = tk.Frame(context_frame)
        context_inner.pack(fill='x', expand=True)

        for label, value in contexts:
            tk.Radiobutton(
                context_inner, text=label,
                variable=self.context_var, value=value
            ).pack(side='left', padx=10, expand=True)

        # Boutons
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=5, column=0, pady=15, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        tk.Button(
            button_frame, text="🔍 Analyse Complète",
            command=self.calculate_complete_analysis,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=0, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="📊 Détails Techniques",
            command=self.show_technical_details,
            bg=self.colors['purple'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=1, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Effacer",
            command=self.clear_simple_comparison,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=2, padx=3, sticky='ew')

        # Résultats
        result_frame = tk.LabelFrame(
            scrollable_frame, text="📊 Résultats Complets",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=6, column=0, sticky='ew', pady=10, padx=10)
        result_frame.columnconfigure(0, weight=1)

        self.simple_result_text = scrolledtext.ScrolledText(
            result_frame, height=20, font=('Courier', 9),
            wrap=tk.WORD, state='disabled'
        )
        self.simple_result_text.pack(fill='both', expand=True)

    def calculate_complete_analysis(self):
        """Analyse COMPLÈTE avec toutes les métriques."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Calcul détaillé complet
            result = self.calculator.calculate_distance_detailed(
                phrase1, phrase2)
            similarity = result['jaccard_similarity']
            distance = result['jaccard_distance']

            context = self.context_var.get()
            sim_interp = self.calculator.interpret_similarity(
                similarity, context=context)
            dist_interp = self.calculator.interpret_distance(
                distance, context=context)

            output = f"""
{'='*75}
ANALYSE COMPLÈTE - SIMILARITÉ DE JACCARD
{'='*75}

📝 PHRASES ANALYSÉES:
──────────────────────────────────────────────────────────────────────────
Phrase 1: "{result['sentence1']}"
Phrase 2: "{result['sentence2']}"

⚙️  CONFIGURATION ACTIVE:
──────────────────────────────────────────────────────────────────────────
• Sensibilité à la casse: {'OUI' if self.case_sensitive.get() else 'NON'}
• Suppression ponctuation: {'OUI' if self.remove_punctuation.get() else 'NON'}
• Stop-words retirés: {'OUI' if self.remove_stopwords.get() else 'NON'}
• Stemming appliqué: {'OUI' if self.use_stemming.get() else 'NON'}
• Contexte d'analyse: {context.upper()}

🔤 ANALYSE DES ENSEMBLES DE MOTS:
──────────────────────────────────────────────────────────────────────────
Ensemble 1 ({len(result['words_set1'])} mots):
  {sorted(result['words_set1'])}

Ensemble 2 ({len(result['words_set2'])} mots):
  {sorted(result['words_set2'])}

∩ INTERSECTION ({result['intersection_size']} mots communs):
  {sorted(result['intersection'])}

∪ UNION ({result['union_size']} mots uniques total):
  {sorted(result['union'])}

📊 MÉTRIQUES DE SIMILARITÉ:
──────────────────────────────────────────────────────────────────────────
Score de Similarité: {similarity:.4f} ({similarity*100:.2f}%)
Catégorie: {sim_interp['emoji']} {sim_interp['category']}

💡 Interprétation Générale:
{sim_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{sim_interp['contextual_interpretation']}

📖 Explication Technique:
{sim_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in sim_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
📏 MÉTRIQUES DE DISTANCE:
──────────────────────────────────────────────────────────────────────────
Score de Distance: {distance:.4f} ({distance*100:.2f}%)
Catégorie: {dist_interp['emoji']} {dist_interp['category']}

💡 Interprétation Générale:
{dist_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{dist_interp['contextual_interpretation']}

📖 Explication Technique:
{dist_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in dist_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
✓ VÉRIFICATION MATHÉMATIQUE:
──────────────────────────────────────────────────────────────────────────
Similarité ({similarity:.4f}) + Distance ({distance:.4f}) = {similarity + distance:.4f}
Formule: J(A,B) = |A ∩ B| / |A ∪ B| = {result['intersection_size']}/{result['union_size']} = {similarity:.4f}

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status(
                f"Analyse terminée | Sim: {similarity:.4f} | Dist: {distance:.4f} | Contexte: {context}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def show_technical_details(self):
        """Affiche les détails techniques COMPLETS."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Prétraitement détaillé
            set1 = self.calculator.preprocess_sentence(phrase1)
            set2 = self.calculator.preprocess_sentence(phrase2)

            intersection = set1.intersection(set2)
            union = set1.union(set2)
            diff1 = set1.difference(set2)
            diff2 = set2.difference(set1)

            similarity = len(intersection) / \
                len(union) if len(union) > 0 else 0.0

            output = f"""
{'='*75}
DÉTAILS TECHNIQUES COMPLETS
{'='*75}

📋 PRÉTRAITEMENT:
──────────────────────────────────────────────────────────────────────────
Phrase originale 1: "{phrase1}"
Après prétraitement: {sorted(set1)}

Phrase originale 2: "{phrase2}"
Après prétraitement: {sorted(set2)}

🔢 OPÉRATIONS SUR ENSEMBLES:
──────────────────────────────────────────────────────────────────────────
|A| = {len(set1)} mots
|B| = {len(set2)} mots

|A ∩ B| = {len(intersection)} mots
Intersection: {sorted(intersection)}

|A ∪ B| = {len(union)} mots
Union: {sorted(union)}

|A - B| = {len(diff1)} mots (uniquement dans A)
Différence A-B: {sorted(diff1)}

|B - A| = {len(diff2)} mots (uniquement dans B)
Différence B-A: {sorted(diff2)}

📐 CALCULS MATHÉMATIQUES:
──────────────────────────────────────────────────────────────────────────
Similarité de Jaccard:
J(A,B) = |A ∩ B| / |A ∪ B|
J(A,B) = {len(intersection)} / {len(union)}
J(A,B) = {similarity:.6f}

Distance de Jaccard:
d(A,B) = 1 - J(A,B)
d(A,B) = 1 - {similarity:.6f}
d(A,B) = {1-similarity:.6f}

📊 STATISTIQUES:
──────────────────────────────────────────────────────────────────────────
Taux de chevauchement: {(len(intersection)/max(len(set1), len(set2))*100) if max(len(set1), len(set2)) > 0 else 0:.2f}%
Mots communs/Phrase 1: {(len(intersection)/len(set1)*100) if len(set1) > 0 else 0:.2f}%
Mots communs/Phrase 2: {(len(intersection)/len(set2)*100) if len(set2) > 0 else 0:.2f}%

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status("Détails techniques affichés")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def clear_simple_comparison(self):
        """Efface les champs."""
        self.phrase1_text.delete("1.0", tk.END)
        self.phrase2_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='normal')
        self.simple_result_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='disabled')
        self.update_status("Champs effacés")

    def create_multiple_comparison_tab(self):
        """Onglet de comparaison multiple."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Multiple  ")

        tab.rowconfigure(2, weight=1)
        tab.rowconfigure(4, weight=1)
        tab.columnconfigure(0, weight=1)

        input_frame = tk.LabelFrame(
            tab, text="📝 Ajouter des phrases",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        input_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        input_frame.columnconfigure(0, weight=1)

        tk.Label(input_frame, text="Nouvelle phrase:").grid(
            row=0, column=0, sticky='w', pady=(0, 5))

        phrase_entry_frame = tk.Frame(input_frame)
        phrase_entry_frame.grid(row=1, column=0, sticky='ew')
        phrase_entry_frame.columnconfigure(0, weight=1)

        self.multi_phrase_entry = tk.Entry(
            phrase_entry_frame, font=('Arial', 10))
        self.multi_phrase_entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.multi_phrase_entry.bind('<Return>', lambda e: self.add_phrase())

        tk.Button(
            phrase_entry_frame, text="➕ Ajouter", command=self.add_phrase,
            bg=self.colors['success'], fg='white', font=('Arial', 10, 'bold')
        ).grid(row=0, column=1)

        list_frame = tk.LabelFrame(
            tab, text="📋 Phrases à comparer",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        list_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.phrases_listbox = tk.Listbox(
            list_frame, font=('Arial', 10),
            yscrollcommand=scrollbar.set, selectmode=tk.SINGLE
        )
        self.phrases_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.phrases_listbox.yview)

        button_frame = tk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10, sticky='ew', padx=10)
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        tk.Button(
            button_frame, text="🔍 Comparer Toutes",
            command=self.compare_multiple_phrases,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="➖ Supprimer",
            command=self.remove_phrase,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=1, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Tout Effacer",
            command=self.clear_all_phrases,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=2, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=4, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.multi_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.multi_result_text.pack(fill='both', expand=True)

    def add_phrase(self):
        """Ajoute une phrase."""
        phrase = self.multi_phrase_entry.get().strip()
        if not phrase:
            messagebox.showwarning("Attention", "Veuillez saisir une phrase.")
            return
        if phrase in self.phrases_list:
            messagebox.showinfo("Information", "Cette phrase existe déjà.")
            return
        self.phrases_list.append(phrase)
        self.phrases_listbox.insert(
            tk.END, f"{len(self.phrases_list)}. {phrase}")
        self.multi_phrase_entry.delete(0, tk.END)
        self.update_status(
            f"Phrase ajoutée ({len(self.phrases_list)} phrases)")

    def remove_phrase(self):
        """Supprime la phrase sélectionnée."""
        selection = self.phrases_listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "Attention", "Veuillez sélectionner une phrase.")
            return
        index = selection[0]
        del self.phrases_list[index]
        self.phrases_listbox.delete(0, tk.END)
        for i, phrase in enumerate(self.phrases_list, 1):
            self.phrases_listbox.insert(tk.END, f"{i}. {phrase}")
        self.update_status(
            f"Phrase supprimée ({len(self.phrases_list)} restantes)")

    def clear_all_phrases(self):
        """Efface toutes les phrases."""
        if self.phrases_list:
            if messagebox.askyesno("Confirmation", "Effacer toutes les phrases?"):
                self.phrases_list.clear()
                self.phrases_listbox.delete(0, tk.END)
                self.multi_result_text.config(state='normal')
                self.multi_result_text.delete("1.0", tk.END)
                self.multi_result_text.config(state='disabled')
                self.update_status("Toutes les phrases effacées")

    def compare_multiple_phrases(self):
        """Compare toutes les phrases."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = self.calculator.compare_multiple_sentences(
                self.phrases_list)
            results.sort(key=lambda x: x[2], reverse=True)

            output = f"""
{'='*70}
COMPARAISON MULTIPLE DE PHRASES
{'='*70}

Nombre de phrases: {len(self.phrases_list)}
Nombre de comparaisons: {len(results)}

{'─'*70}
TOP 10 PAIRES LES PLUS SIMILAIRES:
{'─'*70}
"""
            for i, (idx1, idx2, sim) in enumerate(results[:10], 1):
                output += f"\n{i}. Similarité: {sim:.4f}\n"
                output += f"   Phrase {idx1+1}: {self.phrases_list[idx1][:60]}...\n"
                output += f"   Phrase {idx2+1}: {self.phrases_list[idx2][:60]}...\n"

            idx1, idx2, max_sim = self.calculator.get_most_similar_pair(
                self.phrases_list)
            output += f"\n{'─'*70}\n🏆 PAIRE LA PLUS SIMILAIRE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_sim:.4f}\n"

            idx1, idx2, max_dist = self.calculator.get_most_different_pair(
                self.phrases_list)
            output += f"\n📏 PAIRE LA PLUS DIFFÉRENTE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_dist:.4f} (distance)\n"
            output += f"{'='*70}\n"

            self.multi_result_text.config(state='normal')
            self.multi_result_text.delete("1.0", tk.END)
            self.multi_result_text.insert("1.0", output)
            self.multi_result_text.config(state='disabled')

            self.update_status(f"Comparaison: {len(results)} paires analysées")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_matrix_tab(self):
        """Onglet matrices."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Matrices  ")

        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        info_label = tk.Label(
            tab,
            text="Matrices de similarité et distance pour les phrases de 'Comparaison Multiple'.",
            font=('Arial', 10), wraplength=900
        )
        info_label.grid(row=0, column=0, pady=10, padx=20, sticky='w')

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="📊 Matrice Similarité",
            command=lambda: self.generate_matrix('similarity'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Matrice Distance",
            command=lambda: self.generate_matrix('distance'),
            bg=self.colors['primary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=1, padx=5, sticky='ew')

        matrix_frame = tk.LabelFrame(
            tab, text="🔢 Matrice",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        matrix_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        matrix_frame.rowconfigure(0, weight=1)
        matrix_frame.columnconfigure(0, weight=1)

        self.matrix_text = scrolledtext.ScrolledText(
            matrix_frame, font=('Courier', 9), wrap=tk.NONE, state='disabled')
        self.matrix_text.grid(row=0, column=0, sticky='nsew')

        xscrollbar = tk.Scrollbar(matrix_frame, orient='horizontal')
        xscrollbar.grid(row=1, column=0, sticky='ew')
        self.matrix_text.config(xscrollcommand=xscrollbar.set)
        xscrollbar.config(command=self.matrix_text.xview)

    def generate_matrix(self, matrix_type='similarity'):
        """Génère la matrice."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            if matrix_type == 'similarity':
                matrix = self.calculator.get_similarity_matrix(
                    self.phrases_list)
                title = "MATRICE DE SIMILARITÉ"
                legend = "Valeurs élevées = très similaires"
            else:
                matrix = self.calculator.get_distance_matrix(self.phrases_list)
                title = "MATRICE DE DISTANCE"
                legend = "Valeurs élevées = très différents"

            output = f"""
{'='*70}
{title}
{'='*70}

Phrases analysées:
"""
            for i, phrase in enumerate(self.phrases_list):
                output += f"  {i}: {phrase[:60]}...\n"

            output += f"\n{'─'*70}\nMatrice:\n\n     "
            for i in range(len(self.phrases_list)):
                output += f"{i:8}"
            output += "\n"

            for i, row in enumerate(matrix):
                output += f"{i:3}: "
                for value in row:
                    output += f"{value:8.4f}"
                output += "\n"

            output += f"\n{'='*70}\nLégende:\n"
            output += f"  • Diagonale = {'1.00' if matrix_type == 'similarity' else '0.00'}\n"
            output += f"  • {legend}\n"
            output += f"{'='*70}\n"

            self.matrix_text.config(state='normal')
            self.matrix_text.delete("1.0", tk.END)
            self.matrix_text.insert("1.0", output)
            self.matrix_text.config(state='disabled')

            self.update_status(f"Matrice {matrix_type} générée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_extreme_pairs_tab(self):
        """NOUVEAU: Onglet paires extrêmes."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Paires Extrêmes  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="ℹ️ Information",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Recherche automatique des paires les plus similaires et les plus différentes.",
            font=('Arial', 10), wraplength=900
        ).pack()

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="🏆 Paire la Plus Similaire",
            command=self.find_most_similar,
            bg=self.colors['success'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Paire la Plus Différente",
            command=self.find_most_different,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=1, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.extreme_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.extreme_result_text.pack(fill='both', expand=True)

    def find_most_similar(self):
        """Trouve la paire la plus similaire."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, similarity = self.calculator.get_most_similar_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_similarity(similarity)

            output = f"""
{'='*70}
🏆 PAIRE LA PLUS SIMILAIRE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📊 SCORE DE SIMILARITÉ: {similarity:.4f} ({similarity*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus similaire: {similarity:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def find_most_different(self):
        """Trouve la paire la plus différente."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, distance = self.calculator.get_most_different_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_distance(distance)

            output = f"""
{'='*70}
📏 PAIRE LA PLUS DIFFÉRENTE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📏 SCORE DE DISTANCE: {distance:.4f} ({distance*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus différente: {distance:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_demo_tests_tab(self):
        """NOUVEAU: Onglet tests automatiques."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Tests Auto  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="🧪 Tests Automatiques",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Lance des tests de démonstration comme jaccard_similarity.py",
            font=('Arial', 10)
        ).pack()

        tk.Button(
            tab, text="▶️ Lancer Tests de Démonstration",
            command=self.run_demo_tests,
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).grid(row=1, column=0, pady=20)

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats des Tests",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.demo_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.demo_result_text.pack(fill='both', expand=True)

    def run_demo_tests(self):
        """Lance les tests de démonstration."""
        examples = [
            ("Le chat mange des croquettes", "Le chien mange des croquettes"),
            ("Python est un langage de programmation",
             "Java est un langage de programmation"),
            ("Machine learning supervisé", "Apprentissage automatique supervisé"),
            ("Bonjour tout le monde", "Salut tout le monde"),
            ("Aucun mot en commun", "Différentes phrases complètement")
        ]

        output = f"""
{'='*70}
TESTS DE DÉMONSTRATION AUTOMATIQUES
{'='*70}

Configuration active:
  - Sensibilité à la casse: {'Activée' if self.case_sensitive.get() else 'Désactivée'}
  - Stop-words: {'Activés' if self.remove_stopwords.get() else 'Désactivés'}
  - Stemming: {'Activé' if self.use_stemming.get() else 'Désactivé'}

{'─'*70}
TESTS:
{'─'*70}
"""

        for i, (s1, s2) in enumerate(examples, 1):
            similarity = self.calculator.calculate_similarity(s1, s2)
            distance = self.calculator.calculate_distance(s1, s2)

            sim_interp = self.calculator.interpret_similarity(similarity)
            dist_interp = self.calculator.interpret_distance(distance)

            output += f"""
Test {i}:
  Phrase 1: '{s1}'
  Phrase 2: '{s2}'

  📊 SIMILARITÉ: {similarity:.4f}
     Catégorie: {sim_interp['emoji']} {sim_interp['category']}
     {sim_interp['general_interpretation'][:100]}...

  📏 DISTANCE: {distance:.4f}
     Catégorie: {dist_interp['emoji']} {dist_interp['category']}

  ✓ Vérification: {similarity:.4f} + {distance:.4f} = {similarity + distance:.4f}
{'─'*70}
"""

        output += f"\n{'='*70}\nTOUS LES TESTS TERMINÉS\n{'='*70}\n"

        self.demo_result_text.config(state='normal')
        self.demo_result_text.delete("1.0", tk.END)
        self.demo_result_text.insert("1.0", output)
        self.demo_result_text.config(state='disabled')

        self.update_status(f"{len(examples)} tests de démonstration terminés")

    def create_export_tab(self):
        """Onglet export."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Export  ")

        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame, text="Export des Résultats",
            font=('Arial', 16, 'bold')
        ).pack(pady=20)

        tk.Label(
            main_frame,
            text="Exportez les résultats au format CSV ou JSON",
            font=('Arial', 11), wraplength=600
        ).pack(pady=10)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=30)

        tk.Button(
            button_frame, text="💾 Exporter CSV",
            command=lambda: self.export_results('csv'),
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        tk.Button(
            button_frame, text="💾 Exporter JSON",
            command=lambda: self.export_results('json'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        self.export_status = tk.Label(main_frame, text="", font=('Arial', 10))
        self.export_status.pack(pady=20)

    def export_results(self, format_type):
        """Exporte les résultats."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = []
            for i in range(len(self.phrases_list)):
                for j in range(i + 1, len(self.phrases_list)):
                    detailed = self.calculator.calculate_distance_detailed(
                        self.phrases_list[i], self.phrases_list[j])
                    results.append(detailed)

            if format_type == 'csv':
                filename = self.calculator.export_results_to_csv(results)
            else:
                filename = self.calculator.export_results_to_json(results)

            if filename:
                self.export_status.config(
                    text=f"✓ Export réussi: {filename}",
                    fg=self.colors['success']
                )
                messagebox.showinfo("Succès", f"Fichier créé:\n{filename}")
            else:
                self.export_status.config(
                    text="❌ Échec de l'export",
                    fg=self.colors['danger']
                )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_about_tab(self):
        """Onglet À propos."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ℹ️ À Propos  ")

        main_frame = tk.Frame(tab, bg='white')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame,
            text="Calculateur de Similarité de Jaccard",
            font=('Arial', 16, 'bold'), bg='white', fg=self.colors['primary']
        ).pack(pady=10)

        description = """
✨ Version 2.2 - Interface Graphique Complète

Cette interface implémente TOUTES les fonctionnalités de jaccard_similarity.py:

• Calcul de similarité ET distance de Jaccard
• Stemming français amélioré
• Support des stop-words français (60+)
• Interprétation contextuelle (5 contextes)
• Export CSV et JSON
• Tests automatiques de démonstration
• Recherche de paires extrêmes
• Matrices complètes
• Analyse technique détaillée
• Interface flexible et responsive

📐 Formules:
Similarité(A,B) = |A ∩ B| / |A ∪ B|
Distance(A,B) = 1 - Similarité(A,B)
        """

        tk.Label(
            main_frame, text=description, font=('Arial', 10),
            bg='white', justify='left'
        ).pack(pady=20)

        team_frame = tk.LabelFrame(
            main_frame, text="👥 Équipe",
            font=('Arial', 11, 'bold'), bg='white', padx=20, pady=15
        )
        team_frame.pack(fill='x', pady=10)

        for member in ["OUEDRAOGO Lassina", "OUEDRAOGO Rasmane", "POUBERE Abdourazakou"]:
            tk.Label(
                team_frame, text=f"• {member}",
                font=('Arial', 10), bg='white'
            ).pack(anchor='w', pady=2)

        tk.Label(
            main_frame,
            text="📚 Machine Learning non Supervisé\n🎓 Octobre 2025 - Version Complète v2.2",
            font=('Arial', 10), bg='white', fg=self.colors['dark']
        ).pack(pady=20)

    def create_status_bar(self):
        """Crée la barre de statut."""
        self.status_bar = tk.Label(
            self.root,
            text="Prêt | Interface Complète avec TOUTES les fonctionnalités",
            bd=1, relief=tk.SUNKEN, anchor='w', font=('Arial', 9)
        )
        self.status_bar.grid(row=3, column=0, sticky='ew')

    def update_status(self, message):
        """Met à jour le statut."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main():
    """Point d'entrée."""
    root = tk.Tk()
    app = JaccardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

# README.md

```markdown
# Projet Machine Learning non Supervisé - Similarité de Jaccard

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

### 3. Options avancées

```bash
# Prise en compte de la casse
python jaccard_similarity.py --case-sensitive
python jaccard_similarity.py --interactive --case-sensitive

# Suppression des stop-words français
python jaccard_similarity.py --remove-stopwords
python jaccard_similarity.py --interactive --remove-stopwords

# Utilisation du stemming français
python jaccard_similarity.py --use-stemming
python jaccard_similarity.py --interactive --use-stemming

# Combinaison d'options
python jaccard_similarity.py --remove-stopwords --use-stemming
python jaccard_similarity.py --interactive --remove-stopwords --use-stemming

# Export des résultats
python jaccard_similarity.py --export csv
python jaccard_similarity.py --export json
python jaccard_similarity.py --export both
```

### 4. Interface graphique

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

### Exemple 2 : Utilisation des stop-words

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

### Exemple 3 : Utilisation du stemming

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

### Exemple 4 : Combinaison optimale

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

### Exemple 5 : Interprétation contextuelle

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

### Exemple 6 : Export des résultats

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

### Exemple 7 : Analyse détaillée avec distance

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

### Exemple 8 : Matrice de distance

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

### Exemple 9 : Recherche de paires extrêmes

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

##### Calcul de distance

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

Trouve la paire la plus différente.

##### Interprétation

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

##### Export

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

### Stop-words français

Plus de 60 mots courants filtrés :

- Articles : le, la, les, un, une, des, de, du, au, aux
- Pronoms : je, tu, il, elle, on, nous, vous, ils, elles
- Prépositions : à, dans, par, pour, en, vers, avec, sans
- Conjonctions : et, ou, mais, donc, or, ni, car
- Et bien d'autres...

### Stemming français

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

### Tests de performance

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

**Tests de base :**

- Phrases identiques (similarité = 1.0)
- Phrases sans mots communs (similarité = 0.0)
- Cas partiels avec calculs vérifiés
- Gestion de la ponctuation et de la casse
- Chaînes vides et cas limites
- Propriétés mathématiques (réflexivité, symétrie)
- Tests de performance

**Nouveaux tests :**

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

```

```

# QUICKSTART.md

```markdown

```

Voici la version 2.0 du projet :

# jaccard_similarity.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de calcul de similarité de Jaccard entre phrases
Projet de Machine Learning non Supervisé - VERSION v2.0

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

La similarité de Jaccard mesure la ressemblance entre deux ensembles
en calculant le rapport entre l'intersection et l'union des ensembles.
Formule: Jaccard(A,B) = |A ∩ B| / |A ∪ B|
"""

# ============================================================================
# IMPORTS
# ============================================================================
import re  # Pour le nettoyage de texte avec les regex
import argparse  # Pour gérer les arguments en ligne de commande
import json  # Pour l'export JSON
import csv  # Pour l'export CSV
from typing import Set, List, Tuple, Dict  # Pour les annotations de type
from datetime import datetime  # Pour les timestamps dans les exports


# ============================================================================
# CLASSE FrenchStemmer
# Cette classe réduit les mots français à leur racine
# Exemple: "manger", "mange", "mangé" deviennent tous "mang"
# ============================================================================
class FrenchStemmer:
    """Stemmer pour le français avec gestion des cas spéciaux."""

    # Liste des suffixes français, triés du plus long au plus court
    # Important: l'ordre évite de couper trop tôt (ex: "ation" avant "s")
    SUFFIXES = [
        'issements', 'issement',
        'atrice', 'ations', 'ation', 'atrices',
        'erions', 'eraient', 'assent', 'assiez', 'èrent',
        'erons', 'eront', 'erait', 'eriez', 'erais',
        'ements', 'ement', 'euses', 'euse', 'istes', 'iste',
        'ables', 'able', 'ances', 'ance', 'ences', 'ence',
        'ments', 'ment', 'ités', 'ité', 'eurs', 'eur',
        'eaux', 'aux', 'ant', 'ent', 'ait', 'ais',
        'er', 'es', 'é', 'ée', 'és', 'ées', 's'
    ]

    # Mots qu'on ne doit jamais stemmer
    # Ce sont des mots grammaticaux qui perdraient leur sens
    PROTECTED_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'de', 'du', 'au', 'aux', 'ce', 'ces',
        'et', 'ou', 'mais', 'car', 'or', 'donc', 'ni',
        'si', 'ne', 'pas', 'plus', 'très', 'bien', 'tout'
    }

    # Dictionnaire des verbes irréguliers
    # On associe les formes conjuguées à une racine commune
    EXCEPTIONS = {
        'suis': 'êtr', 'es': 'êtr', 'est': 'êtr',
        'sommes': 'êtr', 'êtes': 'êtr', 'sont': 'êtr',
        'ai': 'av', 'as': 'av', 'a': 'av',
        'avons': 'av', 'avez': 'av', 'ont': 'av',
        'vais': 'all', 'va': 'all', 'allons': 'all',
        'allez': 'all', 'vont': 'all',
    }

    @staticmethod
    def stem(word: str) -> str:
        """
        Applique le stemming à un mot français.

        Paramètres:
            word (str): Le mot à traiter

        Retourne:
            str: La racine du mot

        Logique:
            On suit plusieurs étapes pour décider comment traiter le mot
        """
        # Les mots trop courts (≤2 caractères) sont laissés tels quels
        if len(word) <= 2:
            return word.lower()

        word_lower = word.lower()

        # Vérifier si c'est un mot protégé
        if word_lower in FrenchStemmer.PROTECTED_WORDS:
            return word_lower

        # Vérifier si c'est une forme irrégulière connue
        if word_lower in FrenchStemmer.EXCEPTIONS:
            return FrenchStemmer.EXCEPTIONS[word_lower]

        # Essayer d'enlever un suffixe
        for suffix in FrenchStemmer.SUFFIXES:
            if word_lower.endswith(suffix):
                stem_candidate = word_lower[:-len(suffix)]
                # On garde la racine seulement si elle fait au moins 3 caractères
                if len(stem_candidate) >= 3:
                    return stem_candidate

        # Si aucune règle ne marche, on retourne le mot en minuscules
        return word_lower


# ============================================================================
# FONCTIONS DE VALIDATION
# Ces fonctions vérifient que les données sont correctes avant traitement
# ============================================================================

def validate_sentence(sentence: str, allow_empty: bool = False) -> bool:
    """
    Valide une phrase avant de la traiter.

    Paramètres:
        sentence (str): La phrase à valider
        allow_empty (bool): Autorise les phrases vides si True

    Retourne:
        bool: True si tout est bon

    Lève une exception si:
        - Ce n'est pas une chaîne de caractères
        - La phrase est vide (sauf si allow_empty=True)
        - La phrase est trop longue
    """
    # Vérifier le type
    if not isinstance(sentence, str):
        raise TypeError(
            f"La phrase doit être une chaîne de caractères, pas {type(sentence).__name__}")

    # Vérifier si vide
    if not allow_empty and not sentence.strip():
        raise ValueError("La phrase ne peut pas être vide")

    # Vérifier la longueur (limite à 10000 caractères)
    if len(sentence) > 10000:
        raise ValueError(
            f"La phrase est trop longue ({len(sentence)} caractères, max 10000)")

    return True


def validate_sentences_list(sentences: List[str], min_length: int = 2) -> bool:
    """
    Valide une liste de phrases.

    Paramètres:
        sentences (List[str]): Liste à valider
        min_length (int): Nombre minimum de phrases requis

    Retourne:
        bool: True si tout est bon

    Cette fonction est utilisée avant les opérations sur plusieurs phrases
    (comme les matrices ou les comparaisons multiples)
    """
    # Vérifier que c'est bien une liste
    if not isinstance(sentences, list):
        raise TypeError(
            f"sentences doit être une liste, pas {type(sentences).__name__}")

    # Vérifier qu'il y a assez de phrases
    if len(sentences) < min_length:
        raise ValueError(
            f"Au moins {min_length} phrases sont requises, {len(sentences)} fournies")

    # Vérifier chaque phrase individuellement
    for i, sentence in enumerate(sentences):
        try:
            validate_sentence(sentence)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Phrase {i} invalide: {e}")

    return True


# ============================================================================
# CLASSE JaccardSimilarity
# C'est la classe principale qui fait tous les calculs
# ============================================================================
class JaccardSimilarity:
    """Classe pour calculer la similarité de Jaccard entre phrases."""

    # Liste des stop-words français (mots très fréquents qui apportent peu de sens)
    FRENCH_STOPWORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
        'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
        'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'lui', 'y', 'en',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
        'à', 'dans', 'par', 'pour', 'en', 'vers', 'avec', 'sans', 'sous', 'sur',
        'qui', 'que', 'quoi', 'dont', 'où',
        'si', 'ne', 'pas', 'plus', 'moins', 'très', 'tout', 'toute', 'tous', 'toutes',
        'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
        'falloir', 'vouloir', 'venir', 'devoir', 'croire', 'trouver', 'donner',
        'prendre', 'parler', 'aimer', 'passer', 'mettre'
    }

    def __init__(self, case_sensitive: bool = False, remove_punctuation: bool = True,
                 remove_stopwords: bool = False, use_stemming: bool = False):
        """
        Initialise le calculateur avec les options choisies.

        Paramètres:
            case_sensitive: Si True, "Python" et "python" sont différents
            remove_punctuation: Si True, enlève la ponctuation
            remove_stopwords: Si True, filtre les stop-words
            use_stemming: Si True, applique le stemming
        """
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

        # On crée le stemmer seulement si on en a besoin
        self.stemmer = FrenchStemmer() if use_stemming else None

    def preprocess_sentence(self, sentence: str) -> Set[str]:
        """
        Prétraite une phrase et la convertit en ensemble de mots.

        Paramètre:
            sentence (str): La phrase à traiter

        Retourne:
            Set[str]: Ensemble des mots (sans doublons)

        Étapes du traitement:
            1. Normalisation de la casse
            2. Suppression de la ponctuation
            3. Découpage en mots
            4. Filtrage des stop-words
            5. Stemming

        On utilise un Set parce que Jaccard travaille sur des ensembles,
        donc les répétitions ne comptent pas
        """
        # Normalisation de la casse
        if not self.case_sensitive:
            sentence = sentence.lower()

        # Suppression de la ponctuation avec une regex
        # On garde seulement les lettres et les espaces
        if self.remove_punctuation:
            sentence = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', sentence)

        # Découpage en mots
        words = [word.strip() for word in sentence.split() if word.strip()]

        # Filtrage des stop-words si activé
        if self.remove_stopwords:
            words = [w for w in words if w.lower() not in self.FRENCH_STOPWORDS]

        # Application du stemming si activé
        if self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(w) for w in words]

        # Conversion en Set
        return set(words)

    def calculate_similarity_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la similarité de Jaccard avec tous les détails.

        Paramètres:
            sentence1, sentence2: Les deux phrases à comparer

        Retourne:
            Dict: Dictionnaire avec:
                - Les phrases originales
                - Les ensembles de mots
                - L'intersection et l'union
                - Le score de similarité

        Formule: Similarité = |A ∩ B| / |A ∪ B|
        """
        # Validation
        validate_sentence(sentence1, allow_empty=True)
        validate_sentence(sentence2, allow_empty=True)

        # Prétraitement
        set1 = self.preprocess_sentence(sentence1)
        set2 = self.preprocess_sentence(sentence2)

        # Calcul de l'intersection (mots communs)
        intersection = set1.intersection(set2)

        # Calcul de l'union (tous les mots uniques)
        union = set1.union(set2)

        # Calcul de la similarité
        # Si l'union est vide, on retourne 0
        similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'words_set1': set1,
            'words_set2': set2,
            'intersection': intersection,
            'union': union,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'jaccard_similarity': similarity
        }

    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Version simple qui retourne juste le score de similarité.

        Retourne un float entre 0 et 1:
            - 0 = aucun mot commun
            - 1 = phrases identiques
        """
        validate_sentence(sentence1, allow_empty=True)
        validate_sentence(sentence2, allow_empty=True)

        result = self.calculate_similarity_detailed(sentence1, sentence2)
        return result['jaccard_similarity']

    def calculate_distance_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la distance de Jaccard avec tous les détails.

        La distance est le complément de la similarité: Distance = 1 - Similarité

        Propriétés de la distance:
            - Distance entre phrases identiques = 0
            - Distance entre phrases sans mots communs = 1
            - La distance respecte l'inégalité triangulaire
        """
        detailed = self.calculate_similarity_detailed(sentence1, sentence2)
        detailed['jaccard_distance'] = 1.0 - detailed['jaccard_similarity']
        return detailed

    def calculate_distance(self, sentence1: str, sentence2: str) -> float:
        """
        Version simple qui retourne juste le score de distance.
        """
        result = self.calculate_distance_detailed(sentence1, sentence2)
        return result['jaccard_distance']

    def compare_multiple_sentences(self, sentences: List[str]) -> List[Tuple[int, int, float]]:
        """
        Compare toutes les paires possibles dans une liste de phrases.

        Paramètre:
            sentences: Liste des phrases

        Retourne:
            Liste de tuples (index1, index2, similarité)

        Pour n phrases, on fait n(n-1)/2 comparaisons
        Exemple: 4 phrases → 6 comparaisons

        Complexité: O(n²)
        """
        validate_sentences_list(sentences, min_length=2)

        results = []

        # Double boucle pour générer toutes les paires
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self.calculate_similarity(
                    sentences[i], sentences[j])
                results.append((i, j, similarity))

        return results

    def get_metric_matrix(self, sentences: List[str], metric: str = 'similarity') -> List[List[float]]:
        """
        Calcule une matrice de similarité ou de distance.

        Paramètres:
            sentences: Liste des phrases
            metric: 'similarity' ou 'distance'

        Retourne:
            Matrice n×n (liste de listes)

        La matrice est symétrique, et la diagonale contient:
            - 1.0 pour la similarité (une phrase est identique à elle-même)
            - 0.0 pour la distance

        Utile pour les algorithmes de clustering, les visualisations, etc.
        """
        validate_sentences_list(sentences, min_length=1)

        n = len(sentences)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        # Choix de la fonction selon la métrique
        calc_func = self.calculate_similarity if metric == 'similarity' else self.calculate_distance
        diagonal_value = 1.0 if metric == 'similarity' else 0.0

        # Remplissage de la matrice
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = diagonal_value
                else:
                    matrix[i][j] = calc_func(sentences[i], sentences[j])

        return matrix

    def get_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """Raccourci pour calculer la matrice de similarité."""
        return self.get_metric_matrix(sentences, 'similarity')

    def get_distance_matrix(self, sentences: List[str]) -> List[List[float]]:
        """Raccourci pour calculer la matrice de distance."""
        return self.get_metric_matrix(sentences, 'distance')

    def get_extreme_pair(self, sentences: List[str], mode: str = 'most_similar') -> Tuple[int, int, float]:
        """
        Trouve la paire de phrases la plus similaire ou la plus différente.

        Paramètres:
            sentences: Liste des phrases
            mode: 'most_similar' ou 'most_different'

        Retourne:
            Tuple (index1, index2, score)

        Cas d'usage:
            - Détection de plagiat (most_similar)
            - Analyse de diversité (most_different)
        """
        validate_sentences_list(sentences, min_length=2)

        if mode == 'most_similar':
            comparisons = self.compare_multiple_sentences(sentences)
            if not comparisons:
                return (0, 0, 0.0)
            return max(comparisons, key=lambda x: x[2])
        else:
            # Pour la distance max, on fait une recherche manuelle
            max_distance = -1
            max_pair = (0, 0)

            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    distance = self.calculate_distance(
                        sentences[i], sentences[j])
                    if distance > max_distance:
                        max_distance = distance
                        max_pair = (i, j)

            return (max_pair[0], max_pair[1], max_distance)

    def get_most_similar_pair(self, sentences: List[str]) -> Tuple[int, int, float]:
        """Trouve la paire la plus similaire."""
        return self.get_extreme_pair(sentences, 'most_similar')

    def get_most_different_pair(self, sentences: List[str]) -> Tuple[int, int, float]:
        """Trouve la paire la plus différente."""
        return self.get_extreme_pair(sentences, 'most_different')

    def interpret_metric(self, value: float, metric: str = 'similarity', context: str = 'general') -> Dict[str, str]:
        """
        Interprète un score en fonction du contexte.

        Paramètres:
            value: Le score à interpréter
            metric: 'similarity' ou 'distance'
            context: Contexte d'utilisation (plagiarism, clustering, search, diversity, general)

        Retourne:
            Dict avec:
                - score: La valeur numérique
                - category: Catégorie ("Très similaire", etc.)
                - emoji: Émoji visuel
                - color_code: Code couleur
                - general_interpretation: Explication générale
                - contextual_interpretation: Explication selon le contexte
                - recommendations: Recommandations
                - technical_explanation: Explication mathématique

        Catégories:
            1.0 → Identique
            ≥0.8 → Très similaire
            ≥0.6 → Assez similaire
            ≥0.4 → Moyennement similaire
            ≥0.2 → Peu similaire
            >0 → Très peu similaire
            0 → Aucune similarité
        """
        # Conversion entre distance et similarité
        if metric == 'distance':
            similarity = 1.0 - value
            distance = value
        else:
            similarity = value
            distance = 1.0 - value

        # Catégorisation du score
        if similarity == 1.0:
            category = "Identique"
            emoji = "✅"
            color_code = "green"
        elif similarity >= 0.8:
            category = "Très similaire"
            emoji = "🟢"
            color_code = "green"
        elif similarity >= 0.6:
            category = "Assez similaire"
            emoji = "🟡"
            color_code = "yellow"
        elif similarity >= 0.4:
            category = "Moyennement similaire" if metric == 'similarity' else "Moyennement différent"
            emoji = "🟠"
            color_code = "orange"
        elif similarity >= 0.2:
            category = "Peu similaire" if metric == 'similarity' else "Très différent"
            emoji = "🔴"
            color_code = "red"
        elif similarity > 0:
            category = "Très peu similaire"
            emoji = "⚫"
            color_code = "dark_red"
        else:
            category = "Aucune similarité" if metric == 'similarity' else "Complètement différent"
            emoji = "❌"
            color_code = "black"

        # Construction du résultat
        result = {
            'score': value,
            'category': category,
            'emoji': emoji,
            'color_code': color_code,
            'general_interpretation': self._get_unified_general_interpretation(similarity),
            'contextual_interpretation': self._get_unified_contextual_interpretation(similarity, context, metric),
            'recommendations': self._get_unified_recommendations(similarity, context, metric),
            'technical_explanation': self._get_unified_technical_explanation(similarity, distance, metric)
        }

        if metric == 'distance':
            result['similarity'] = similarity
            result['distance'] = distance

        return result

    def interpret_similarity(self, similarity: float, context: str = "general") -> Dict[str, str]:
        """Raccourci pour interpréter une similarité."""
        return self.interpret_metric(similarity, 'similarity', context)

    def interpret_distance(self, distance: float, context: str = "general") -> Dict[str, str]:
        """Raccourci pour interpréter une distance."""
        return self.interpret_metric(distance, 'distance', context)

    def _get_unified_general_interpretation(self, similarity: float) -> str:
        """
        Fournit une explication générale du score.

        Traduit le score numérique en texte compréhensible
        """
        if similarity == 1.0:
            return ("Les deux phrases sont parfaitement identiques. Tous les mots sont communs "
                    "et aucun mot unique n'existe dans l'une ou l'autre phrase.")
        elif similarity >= 0.8:
            return ("Les phrases partagent la grande majorité de leurs mots. Elles expriment "
                    "probablement des idées très proches avec une formulation similaire.")
        elif similarity >= 0.6:
            return ("Les phrases ont une base commune importante mais contiennent aussi des "
                    "différences notables. Elles traitent probablement du même sujet mais "
                    "avec des nuances.")
        elif similarity >= 0.4:
            return ("Les phrases partagent certains mots-clés mais diffèrent sensiblement. "
                    "Elles peuvent traiter de sujets connexes ou utiliser un vocabulaire commun "
                    "dans des contextes différents.")
        elif similarity >= 0.2:
            return ("Les phrases ont quelques mots en commun, probablement des mots fréquents "
                    "ou génériques. Elles sont globalement différentes dans leur contenu.")
        elif similarity > 0:
            return ("Les phrases partagent très peu de mots. Il peut s'agir de mots très "
                    "courants (articles, prépositions) sans lien sémantique fort.")
        else:
            return ("Aucun mot n'est partagé entre les deux phrases. Elles traitent de "
                    "sujets complètement différents ou utilisent des vocabulaires distincts.")

    def _get_unified_contextual_interpretation(self, similarity: float, context: str, metric: str) -> str:
        """
        Fournit une interprétation adaptée au contexte.

        Chaque contexte a ses propres seuils:
            - Plagiarism: score élevé = alerte
            - Clustering: score moyen = même groupe
            - Search: score élevé = pertinent
            - Diversity: score faible = varié
        """
        # Dictionnaire d'interprétations par contexte
        interpretations = {
            'plagiarism': {
                1.0: "🚨 PLAGIAT CERTAIN - Copie intégrale détectée",
                0.8: "⚠️  PLAGIAT TRÈS PROBABLE - Similarité suspecte, nécessite une vérification",
                0.6: "⚠️  SUSPICION ÉLEVÉE - Peut indiquer une paraphrase ou réarrangement",
                0.4: "⚡ SUSPICION MODÉRÉE - Quelques éléments communs, à examiner",
                0.2: "✓ SUSPICION FAIBLE - Probablement du contenu original",
                0.0: "✓ CONTENU ORIGINAL - Aucune similarité détectée"
            },
            'clustering': {
                1.0: "📂 CLUSTER IDENTIQUE - Documents identiques ou doublons",
                0.8: "📂 CLUSTER FORT - Documents très liés, même catégorie",
                0.6: "📂 CLUSTER MODÉRÉ - Documents connexes, possiblement même thème",
                0.4: "📂 CLUSTER FAIBLE - Quelques liens, catégories voisines possibles",
                0.2: "📂 PAS DE CLUSTER - Documents distincts",
                0.0: "📂 TOTALEMENT DISTINCTS - Aucun lien apparent"
            },
            'search': {
                1.0: "🎯 PERTINENCE MAXIMALE - Correspondance parfaite avec la requête",
                0.8: "🎯 TRÈS PERTINENT - Contient la plupart des termes de recherche",
                0.6: "🎯 PERTINENT - Bon match avec plusieurs termes clés",
                0.4: "🎯 PARTIELLEMENT PERTINENT - Contient quelques termes de recherche",
                0.2: "🎯 PEU PERTINENT - Match faible avec la requête",
                0.0: "🎯 NON PERTINENT - Aucun terme de recherche trouvé"
            },
            'diversity': {
                1.0: "🎨 AUCUNE DIVERSITÉ - Contenu identique" if metric == 'distance' else "🎨 IDENTIQUE",
                0.8: "🎨 FAIBLE DIVERSITÉ - Contenu très homogène" if metric == 'distance' else "🎨 TRÈS SIMILAIRE",
                0.6: "🎨 DIVERSITÉ MODÉRÉE - Mix de similarités" if metric == 'distance' else "🎨 ASSEZ SIMILAIRE",
                0.4: "🎨 BONNE DIVERSITÉ - Contenus variés" if metric == 'distance' else "🎨 MOYENNEMENT SIMILAIRE",
                0.2: "🎨 FORTE DIVERSITÉ - Contenus très différents" if metric == 'distance' else "🎨 PEU SIMILAIRE",
                0.0: "🎨 DIVERSITÉ MAXIMALE - Contenus totalement distincts" if metric == 'distance' else "🎨 AUCUNE SIMILARITÉ"
            },
            'general': {
                1.0: "Les phrases sont identiques",
                0.8: "Très haute similarité - Contenu très proche",
                0.6: "Bonne similarité - Sujet probablement commun",
                0.4: "Similarité modérée - Quelques éléments partagés",
                0.2: "Faible similarité - Peu d'éléments communs",
                0.0: "Aucune similarité détectée"
            }
        }

        context_interp = interpretations.get(
            context, interpretations['general'])

        # Sélection selon le score
        if similarity == 1.0:
            return context_interp[1.0]
        elif similarity >= 0.8:
            return context_interp[0.8]
        elif similarity >= 0.6:
            return context_interp[0.6]
        elif similarity >= 0.4:
            return context_interp[0.4]
        elif similarity >= 0.2:
            return context_interp[0.2]
        else:
            return context_interp[0.0]

    def _get_unified_recommendations(self, similarity: float, context: str, metric: str) -> List[str]:
        """
        Fournit des recommandations selon le score et le contexte.
        """
        recommendations = []

        if context == 'plagiarism':
            if similarity >= 0.8:
                recommendations.extend([
                    "Vérifier manuellement le document source",
                    "Comparer les citations et références",
                    "Utiliser des outils de détection plus avancés",
                    "Contacter l'auteur pour clarification"
                ])
            elif similarity >= 0.5:
                recommendations.extend([
                    "Examiner les passages spécifiques similaires",
                    "Vérifier si une paraphrase est appropriée",
                    "S'assurer que les sources sont citées"
                ])

        elif context == 'clustering':
            if similarity >= 0.6:
                recommendations.extend([
                    "Regrouper ces documents dans le même cluster",
                    "Analyser les thèmes communs pour mieux les catégoriser"
                ])
            elif similarity >= 0.3:
                recommendations.append(
                    "Considérer comme potentiellement liés, vérifier manuellement")
            else:
                recommendations.append("Séparer dans des clusters différents")

        elif context == 'search':
            if similarity >= 0.4:
                recommendations.append(
                    "Document pertinent, à inclure dans les résultats")
            else:
                recommendations.append(
                    "Document peu pertinent, peut être exclu des résultats")

        elif context == 'diversity':
            if metric == 'distance':
                if similarity <= 0.3:
                    recommendations.append(
                        "Bonne diversité détectée - Contenu varié")
                else:
                    recommendations.append(
                        "Diversité faible - Envisager d'ajouter du contenu différent")

        if similarity == 0.0:
            recommendations.append(
                "Aucun mot commun - Vérifier le prétraitement des textes" if metric == 'similarity'
                else "Aucun mot commun - Vocabulaires totalement différents")
        elif similarity < 0.3 and len(recommendations) == 0:
            recommendations.append(
                "Similarité faible - Ces textes traitent probablement de sujets différents")

        return recommendations if recommendations else ["Aucune recommandation spécifique"]

    def _get_unified_technical_explanation(self, similarity: float, distance: float, metric: str) -> str:
        """
        Fournit une explication technique du score.
        """
        if metric == 'similarity':
            percentage = similarity * 100
            explanation = f"Score de Jaccard: {similarity:.4f} ({percentage:.2f}%)\n\n"

            if similarity == 1.0:
                explanation += ("L'intersection des ensembles de mots égale leur union. "
                                "Mathématiquement: |A ∩ B| = |A ∪ B|")
            elif similarity >= 0.5:
                explanation += (f"Environ {percentage:.0f}% des mots de l'union sont partagés. "
                                f"Cela signifie qu'environ {100-percentage:.0f}% des mots sont uniques "
                                f"à l'une ou l'autre phrase.")
            else:
                explanation += (f"Seulement {percentage:.0f}% des mots de l'union sont communs. "
                                f"La majorité ({100-percentage:.0f}%) des mots sont spécifiques "
                                f"à chaque phrase.")
        else:
            percentage_diff = distance * 100
            explanation = f"Distance de Jaccard: {distance:.4f} ({percentage_diff:.2f}%)\n"
            explanation += f"Similarité correspondante: {similarity:.4f}\n\n"

            if distance == 0.0:
                explanation += "Distance nulle → Ensembles identiques\n"
                explanation += "Mathématiquement: d(A,B) = 1 - |A ∩ B|/|A ∪ B| = 0"
            elif distance == 1.0:
                explanation += "Distance maximale → Ensembles disjoints\n"
                explanation += "Mathématiquement: |A ∩ B| = 0"
            else:
                explanation += (f"Environ {percentage_diff:.0f}% de dissimilarité entre les ensembles.\n"
                                f"Cela signifie {100-percentage_diff:.0f}% de mots sont partagés.")

        return explanation

    def export_results_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        Exporte les résultats au format CSV.

        Paramètres:
            results: Liste des résultats à exporter
            filename: Nom du fichier (auto-généré si None)

        Retourne:
            str: Nom du fichier créé (ou None si erreur)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jaccard_results_{timestamp}.csv"

        if not results:
            print("Aucun résultat à exporter.")
            return None

        fieldnames = ['sentence1', 'sentence2',
                      'similarity', 'distance', 'category']

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow({
                        'sentence1': result.get('sentence1', ''),
                        'sentence2': result.get('sentence2', ''),
                        'similarity': result.get('jaccard_similarity', 0.0),
                        'distance': result.get('jaccard_distance', 1.0),
                        'category': result.get('category', 'N/A')
                    })

            print(f"✓ Résultats exportés vers: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Erreur lors de l'export CSV: {e}")
            return None

    def export_results_to_json(self, results: List[Dict], filename: str = None) -> str:
        """
        Exporte les résultats au format JSON.

        Paramètres:
            results: Liste des résultats
            filename: Nom du fichier (auto-généré si None)

        Retourne:
            str: Nom du fichier créé (ou None si erreur)

        Le fichier JSON contient:
            - timestamp: Date et heure de l'export
            - config: Configuration utilisée
            - results: Les résultats proprement dits
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jaccard_results_{timestamp}.json"

        if not results:
            print("Aucun résultat à exporter.")
            return None

        try:
            # Conversion des Sets en Lists (JSON ne supporte pas les sets)
            export_data = []
            for result in results:
                export_item = result.copy()

                if 'words_set1' in export_item:
                    export_item['words_set1'] = list(export_item['words_set1'])
                if 'words_set2' in export_item:
                    export_item['words_set2'] = list(export_item['words_set2'])
                if 'intersection' in export_item:
                    export_item['intersection'] = list(
                        export_item['intersection'])
                if 'union' in export_item:
                    export_item['union'] = list(export_item['union'])

                export_data.append(export_item)

            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'case_sensitive': self.case_sensitive,
                        'remove_punctuation': self.remove_punctuation,
                        'remove_stopwords': self.remove_stopwords,
                        'use_stemming': self.use_stemming
                    },
                    'results': export_data
                }, jsonfile, indent=2, ensure_ascii=False)

            print(f"✓ Résultats exportés vers: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Erreur lors de l'export JSON: {e}")
            return None


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def run_example_tests(calculator: JaccardSimilarity, export_format: str = None):
    """
    Exécute des tests avec des exemples prédéfinis.

    Cette fonction montre le fonctionnement du programme avec 5 exemples
    qui couvrent différents cas de figure
    """
    print("=== Programme de Calcul de Similarité de Jaccard ===\n")

    print("Configuration active:")
    print(
        f"  - Sensibilité à la casse: {'Activée' if calculator.case_sensitive else 'Désactivée'}")
    print(
        f"  - Stop-words: {'Activés' if calculator.remove_stopwords else 'Désactivés'}")
    print(
        f"  - Stemming: {'Activé' if calculator.use_stemming else 'Désactivé'}")
    print()

    examples = [
        ("Le chat mange des croquettes", "Le chien mange des croquettes"),
        ("Python est un langage de programmation",
         "Java est un langage de programmation"),
        ("Machine learning supervisé", "Apprentissage automatique supervisé"),
        ("Bonjour tout le monde", "Salut tout le monde"),
        ("Aucun mot en commun", "Différentes phrases complètement")
    ]

    print("1. Tests de base avec double analyse (Similarité + Distance) :")
    print("-" * 80)

    all_results = []

    for i, (s1, s2) in enumerate(examples, 1):
        similarity = calculator.calculate_similarity(s1, s2)
        distance = calculator.calculate_distance(s1, s2)

        sim_interpretation = calculator.interpret_similarity(
            similarity, context='general')
        dist_interpretation = calculator.interpret_distance(
            distance, context='general')

        detailed = calculator.calculate_distance_detailed(s1, s2)
        detailed['category'] = sim_interpretation['category']
        all_results.append(detailed)

        print(f"\nTest {i}:")
        print(f"  Phrase 1: '{s1}'")
        print(f"  Phrase 2: '{s2}'")
        print(f"\n  📊 SIMILARITÉ: {similarity:.4f}")
        print(
            f"     Catégorie: {sim_interpretation['emoji']} {sim_interpretation['category']}")
        print(f"     {sim_interpretation['general_interpretation']}")
        print(f"\n  📏 DISTANCE: {distance:.4f}")
        print(
            f"     Catégorie: {dist_interpretation['emoji']} {dist_interpretation['category']}")
        print(f"     {dist_interpretation['general_interpretation']}")
        print(
            f"\n  ✓ Vérification: Similarité + Distance = {similarity + distance:.4f}")
        print("-" * 80)

    # Export si demandé
    if export_format:
        if export_format.lower() == 'csv':
            calculator.export_results_to_csv(all_results)
        elif export_format.lower() == 'json':
            calculator.export_results_to_json(all_results)
        elif export_format.lower() == 'both':
            calculator.export_results_to_csv(all_results)
            calculator.export_results_to_json(all_results)


def interactive_mode(calculator: JaccardSimilarity):
    """
    Mode interactif pour saisir des phrases manuellement.

    L'utilisateur peut saisir ses propres phrases et voir les résultats.
    Taper 'quit' pour sortir.
    """
    print("=== Mode Interactif - Calculateur de Jaccard ===")
    print(f"Configuration: case_sensitive={calculator.case_sensitive}, "
          f"remove_stopwords={calculator.remove_stopwords}, "
          f"use_stemming={calculator.use_stemming}")
    print("Entrez 'quit' pour quitter\n")

    while True:
        sentence1 = input("Phrase 1: ").strip()
        if sentence1.lower() == 'quit':
            break

        sentence2 = input("Phrase 2: ").strip()
        if sentence2.lower() == 'quit':
            break

        try:
            similarity = calculator.calculate_similarity(sentence1, sentence2)
            distance = calculator.calculate_distance(sentence1, sentence2)

            sim_interpretation = calculator.interpret_similarity(
                similarity, context='general')
            dist_interpretation = calculator.interpret_distance(
                distance, context='general')

            print(f"\n{'='*70}")
            print(f"RÉSULTAT DE LA COMPARAISON")
            print(f"{'='*70}")

            set1 = calculator.preprocess_sentence(sentence1)
            set2 = calculator.preprocess_sentence(sentence2)
            intersection = set1.intersection(set2)

            print(
                f"\nMots communs: {sorted(intersection)} ({len(intersection)} mots)")
            print(f"Total mots uniques: {len(set1.union(set2))} mots")

            print(f"\n{'─'*70}")
            print(f"📊 SIMILARITÉ DE JACCARD")
            print(f"{'─'*70}")
            print(f"Score: {similarity:.4f}")
            print(
                f"Catégorie: {sim_interpretation['emoji']} {sim_interpretation['category']}")
            print(f"\n💡 Interprétation:")
            print(f"   {sim_interpretation['general_interpretation']}")

            print(f"\n{'─'*70}")
            print(f"📏 DISTANCE DE JACCARD")
            print(f"{'─'*70}")
            print(f"Score: {distance:.4f}")
            print(
                f"Catégorie: {dist_interpretation['emoji']} {dist_interpretation['category']}")
            print(f"\n💡 Interprétation:")
            print(f"   {dist_interpretation['general_interpretation']}")

            print(
                f"\n✓ Vérification: Similarité ({similarity:.4f}) + Distance ({distance:.4f}) = {similarity + distance:.4f}")
            print("-" * 70)
            print()

        except (ValueError, TypeError) as e:
            print(f"\n❌ Erreur: {e}\n")


def main():
    """
    Fonction principale du programme.

    Gère l'interface en ligne de commande et lance le mode approprié.
    """
    parser = argparse.ArgumentParser(
        description='Calcul de similarité et distance de Jaccard entre phrases',
        epilog='Exemples:\n'
               '  python jaccard_similarity.py\n'
               '  python jaccard_similarity.py --case-sensitive\n'
               '  python jaccard_similarity.py --remove-stopwords --use-stemming\n'
               '  python jaccard_similarity.py --case-sensitive --export both',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--case-sensitive', action='store_true',
                        help='Respecte la casse des mots (Python ≠ python)')
    parser.add_argument('--keep-punctuation', action='store_true',
                        help='Garde la ponctuation')
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='Retire les stop-words français (le, la, les, etc.)')
    parser.add_argument('--use-stemming', action='store_true',
                        help='Applique le stemming français (manger → mang)')
    parser.add_argument('--interactive', action='store_true',
                        help='Mode interactif pour saisir des phrases')
    parser.add_argument('--export', choices=['csv', 'json', 'both'],
                        help='Exporte les résultats (csv, json, ou both)')
    parser.add_argument('--run-tests', action='store_true',
                        help='Exécute les tests unitaires')

    args = parser.parse_args()

    if args.run_tests:
        run_unit_tests()
        return

    calculator = JaccardSimilarity(
        case_sensitive=args.case_sensitive,
        remove_punctuation=not args.keep_punctuation,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming
    )

    if args.interactive:
        interactive_mode(calculator)
    else:
        run_example_tests(calculator, args.export)


def run_unit_tests():
    """
    Placeholder pour les tests unitaires.

    Pour l'instant, cette fonction redirige vers un fichier de tests séparé.
    """
    print("⚠️  Pour exécuter les tests complets, utilisez:")
    print("    python test_jaccard.py")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    main()
```

# jaccard_gui.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique COMPLÈTE pour le calculateur de similarité de Jaccard
Version 2.2 - TOUTES LES FONCTIONNALITÉS

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

Implémente TOUTES les fonctionnalités de jaccard_similarity.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict
import sys
import os
from datetime import datetime

from jaccard_similarity import JaccardSimilarity, FrenchStemmer


class JaccardGUI:
    """Interface graphique complète avec TOUTES les fonctionnalités."""

    def __init__(self, root):
        """Initialise l'interface graphique."""
        self.root = root
        self.root.title("Calculateur de Jaccard v2.2 - Interface Complète")

        self.root.geometry("1300x950")
        self.root.minsize(1000, 750)

        # Configuration responsive
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=0)
        self.root.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'light': '#ECF0F1',
            'dark': '#34495E',
            'purple': '#9B59B6'
        }

        self.calculator = JaccardSimilarity()

        # Variables pour les options
        self.case_sensitive = tk.BooleanVar(value=False)
        self.remove_punctuation = tk.BooleanVar(value=True)
        self.remove_stopwords = tk.BooleanVar(value=False)
        self.use_stemming = tk.BooleanVar(value=False)
        self.context_var = tk.StringVar(value='general')

        self.phrases_list = []
        self.create_widgets()

    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        self.create_header()
        self.create_options_frame()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)

        # Tous les onglets avec TOUTES les fonctionnalités
        self.create_simple_comparison_tab()
        self.create_multiple_comparison_tab()
        self.create_matrix_tab()
        self.create_extreme_pairs_tab()  # NOUVEAU
        self.create_demo_tests_tab()  # NOUVEAU - Tests automatiques
        self.create_export_tab()
        self.create_about_tab()

        self.create_status_bar()

    def create_header(self):
        """Crée l'en-tête."""
        header_frame = tk.Frame(
            self.root, bg=self.colors['primary'], height=80)
        header_frame.grid(row=0, column=0, sticky='ew')
        header_frame.grid_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📊 Calculateur de Jaccard v2.2 - Interface Complète",
            font=('Arial', 20, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="TOUTES les fonctionnalités de jaccard_similarity.py",
            font=('Arial', 10),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        subtitle_label.pack()

    def create_options_frame(self):
        """Crée le cadre des options."""
        options_frame = tk.LabelFrame(
            self.root,
            text="⚙️ Configuration Complète",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        options_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        row1 = tk.Frame(options_frame)
        row1.pack(fill='x', pady=2)

        tk.Checkbutton(
            row1, text="Sensible à la casse",
            variable=self.case_sensitive,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Supprimer ponctuation",
            variable=self.remove_punctuation,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Retirer stop-words",
            variable=self.remove_stopwords,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Stemming français",
            variable=self.use_stemming,
            command=self.update_calculator
        ).pack(side='left', padx=15)

    def update_calculator(self):
        """Met à jour le calculateur."""
        self.calculator = JaccardSimilarity(
            case_sensitive=self.case_sensitive.get(),
            remove_punctuation=self.remove_punctuation.get(),
            remove_stopwords=self.remove_stopwords.get(),
            use_stemming=self.use_stemming.get()
        )
        self.update_status("Configuration mise à jour")

    def create_simple_comparison_tab(self):
        """Onglet de comparaison simple COMPLET."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Simple  ")

        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        scrollbar.grid(row=0, column=1, sticky='ns')

        scrollable_frame.columnconfigure(0, weight=1)

        # Phrase 1
        tk.Label(scrollable_frame, text="Phrase 1:", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase1_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase1_text.grid(row=1, column=0, sticky='ew', pady=5, padx=10)

        # Phrase 2
        tk.Label(scrollable_frame, text="Phrase 2:", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase2_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase2_text.grid(row=3, column=0, sticky='ew', pady=5, padx=10)

        # Contexte
        context_frame = tk.LabelFrame(
            scrollable_frame,
            text="🎯 Contexte d'interprétation",
            font=('Arial', 10, 'bold'),
            padx=10, pady=5
        )
        context_frame.grid(row=4, column=0, sticky='ew', pady=10, padx=10)

        contexts = [
            ('Général', 'general'),
            ('Plagiat', 'plagiarism'),
            ('Clustering', 'clustering'),
            ('Recherche', 'search'),
            ('Diversité', 'diversity')
        ]

        context_inner = tk.Frame(context_frame)
        context_inner.pack(fill='x', expand=True)

        for label, value in contexts:
            tk.Radiobutton(
                context_inner, text=label,
                variable=self.context_var, value=value
            ).pack(side='left', padx=10, expand=True)

        # Boutons
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=5, column=0, pady=15, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        tk.Button(
            button_frame, text="🔍 Analyse Complète",
            command=self.calculate_complete_analysis,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=0, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="📊 Détails Techniques",
            command=self.show_technical_details,
            bg=self.colors['purple'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=1, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Effacer",
            command=self.clear_simple_comparison,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=2, padx=3, sticky='ew')

        # Résultats
        result_frame = tk.LabelFrame(
            scrollable_frame, text="📊 Résultats Complets",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=6, column=0, sticky='ew', pady=10, padx=10)
        result_frame.columnconfigure(0, weight=1)

        self.simple_result_text = scrolledtext.ScrolledText(
            result_frame, height=20, font=('Courier', 9),
            wrap=tk.WORD, state='disabled'
        )
        self.simple_result_text.pack(fill='both', expand=True)

    def calculate_complete_analysis(self):
        """Analyse COMPLÈTE avec toutes les métriques."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Calcul détaillé complet
            result = self.calculator.calculate_distance_detailed(
                phrase1, phrase2)
            similarity = result['jaccard_similarity']
            distance = result['jaccard_distance']

            context = self.context_var.get()
            sim_interp = self.calculator.interpret_similarity(
                similarity, context=context)
            dist_interp = self.calculator.interpret_distance(
                distance, context=context)

            output = f"""
{'='*75}
ANALYSE COMPLÈTE - SIMILARITÉ DE JACCARD
{'='*75}

📝 PHRASES ANALYSÉES:
──────────────────────────────────────────────────────────────────────────
Phrase 1: "{result['sentence1']}"
Phrase 2: "{result['sentence2']}"

⚙️  CONFIGURATION ACTIVE:
──────────────────────────────────────────────────────────────────────────
• Sensibilité à la casse: {'OUI' if self.case_sensitive.get() else 'NON'}
• Suppression ponctuation: {'OUI' if self.remove_punctuation.get() else 'NON'}
• Stop-words retirés: {'OUI' if self.remove_stopwords.get() else 'NON'}
• Stemming appliqué: {'OUI' if self.use_stemming.get() else 'NON'}
• Contexte d'analyse: {context.upper()}

🔤 ANALYSE DES ENSEMBLES DE MOTS:
──────────────────────────────────────────────────────────────────────────
Ensemble 1 ({len(result['words_set1'])} mots):
  {sorted(result['words_set1'])}

Ensemble 2 ({len(result['words_set2'])} mots):
  {sorted(result['words_set2'])}

∩ INTERSECTION ({result['intersection_size']} mots communs):
  {sorted(result['intersection'])}

∪ UNION ({result['union_size']} mots uniques total):
  {sorted(result['union'])}

📊 MÉTRIQUES DE SIMILARITÉ:
──────────────────────────────────────────────────────────────────────────
Score de Similarité: {similarity:.4f} ({similarity*100:.2f}%)
Catégorie: {sim_interp['emoji']} {sim_interp['category']}

💡 Interprétation Générale:
{sim_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{sim_interp['contextual_interpretation']}

📖 Explication Technique:
{sim_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in sim_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
📏 MÉTRIQUES DE DISTANCE:
──────────────────────────────────────────────────────────────────────────
Score de Distance: {distance:.4f} ({distance*100:.2f}%)
Catégorie: {dist_interp['emoji']} {dist_interp['category']}

💡 Interprétation Générale:
{dist_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{dist_interp['contextual_interpretation']}

📖 Explication Technique:
{dist_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in dist_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
✓ VÉRIFICATION MATHÉMATIQUE:
──────────────────────────────────────────────────────────────────────────
Similarité ({similarity:.4f}) + Distance ({distance:.4f}) = {similarity + distance:.4f}
Formule: J(A,B) = |A ∩ B| / |A ∪ B| = {result['intersection_size']}/{result['union_size']} = {similarity:.4f}

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status(
                f"Analyse terminée | Sim: {similarity:.4f} | Dist: {distance:.4f} | Contexte: {context}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def show_technical_details(self):
        """Affiche les détails techniques COMPLETS."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Prétraitement détaillé
            set1 = self.calculator.preprocess_sentence(phrase1)
            set2 = self.calculator.preprocess_sentence(phrase2)

            intersection = set1.intersection(set2)
            union = set1.union(set2)
            diff1 = set1.difference(set2)
            diff2 = set2.difference(set1)

            similarity = len(intersection) / \
                len(union) if len(union) > 0 else 0.0

            output = f"""
{'='*75}
DÉTAILS TECHNIQUES COMPLETS
{'='*75}

📋 PRÉTRAITEMENT:
──────────────────────────────────────────────────────────────────────────
Phrase originale 1: "{phrase1}"
Après prétraitement: {sorted(set1)}

Phrase originale 2: "{phrase2}"
Après prétraitement: {sorted(set2)}

🔢 OPÉRATIONS SUR ENSEMBLES:
──────────────────────────────────────────────────────────────────────────
|A| = {len(set1)} mots
|B| = {len(set2)} mots

|A ∩ B| = {len(intersection)} mots
Intersection: {sorted(intersection)}

|A ∪ B| = {len(union)} mots
Union: {sorted(union)}

|A - B| = {len(diff1)} mots (uniquement dans A)
Différence A-B: {sorted(diff1)}

|B - A| = {len(diff2)} mots (uniquement dans B)
Différence B-A: {sorted(diff2)}

📐 CALCULS MATHÉMATIQUES:
──────────────────────────────────────────────────────────────────────────
Similarité de Jaccard:
J(A,B) = |A ∩ B| / |A ∪ B|
J(A,B) = {len(intersection)} / {len(union)}
J(A,B) = {similarity:.6f}

Distance de Jaccard:
d(A,B) = 1 - J(A,B)
d(A,B) = 1 - {similarity:.6f}
d(A,B) = {1-similarity:.6f}

📊 STATISTIQUES:
──────────────────────────────────────────────────────────────────────────
Taux de chevauchement: {(len(intersection)/max(len(set1), len(set2))*100) if max(len(set1), len(set2)) > 0 else 0:.2f}%
Mots communs/Phrase 1: {(len(intersection)/len(set1)*100) if len(set1) > 0 else 0:.2f}%
Mots communs/Phrase 2: {(len(intersection)/len(set2)*100) if len(set2) > 0 else 0:.2f}%

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status("Détails techniques affichés")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def clear_simple_comparison(self):
        """Efface les champs."""
        self.phrase1_text.delete("1.0", tk.END)
        self.phrase2_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='normal')
        self.simple_result_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='disabled')
        self.update_status("Champs effacés")

    def create_multiple_comparison_tab(self):
        """Onglet de comparaison multiple."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Multiple  ")

        tab.rowconfigure(2, weight=1)
        tab.rowconfigure(4, weight=1)
        tab.columnconfigure(0, weight=1)

        input_frame = tk.LabelFrame(
            tab, text="📝 Ajouter des phrases",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        input_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        input_frame.columnconfigure(0, weight=1)

        tk.Label(input_frame, text="Nouvelle phrase:").grid(
            row=0, column=0, sticky='w', pady=(0, 5))

        phrase_entry_frame = tk.Frame(input_frame)
        phrase_entry_frame.grid(row=1, column=0, sticky='ew')
        phrase_entry_frame.columnconfigure(0, weight=1)

        self.multi_phrase_entry = tk.Entry(
            phrase_entry_frame, font=('Arial', 10))
        self.multi_phrase_entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.multi_phrase_entry.bind('<Return>', lambda e: self.add_phrase())

        tk.Button(
            phrase_entry_frame, text="➕ Ajouter", command=self.add_phrase,
            bg=self.colors['success'], fg='white', font=('Arial', 10, 'bold')
        ).grid(row=0, column=1)

        list_frame = tk.LabelFrame(
            tab, text="📋 Phrases à comparer",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        list_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.phrases_listbox = tk.Listbox(
            list_frame, font=('Arial', 10),
            yscrollcommand=scrollbar.set, selectmode=tk.SINGLE
        )
        self.phrases_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.phrases_listbox.yview)

        button_frame = tk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10, sticky='ew', padx=10)
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        tk.Button(
            button_frame, text="🔍 Comparer Toutes",
            command=self.compare_multiple_phrases,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="➖ Supprimer",
            command=self.remove_phrase,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=1, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Tout Effacer",
            command=self.clear_all_phrases,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=2, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=4, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.multi_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.multi_result_text.pack(fill='both', expand=True)

    def add_phrase(self):
        """Ajoute une phrase."""
        phrase = self.multi_phrase_entry.get().strip()
        if not phrase:
            messagebox.showwarning("Attention", "Veuillez saisir une phrase.")
            return
        if phrase in self.phrases_list:
            messagebox.showinfo("Information", "Cette phrase existe déjà.")
            return
        self.phrases_list.append(phrase)
        self.phrases_listbox.insert(
            tk.END, f"{len(self.phrases_list)}. {phrase}")
        self.multi_phrase_entry.delete(0, tk.END)
        self.update_status(
            f"Phrase ajoutée ({len(self.phrases_list)} phrases)")

    def remove_phrase(self):
        """Supprime la phrase sélectionnée."""
        selection = self.phrases_listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "Attention", "Veuillez sélectionner une phrase.")
            return
        index = selection[0]
        del self.phrases_list[index]
        self.phrases_listbox.delete(0, tk.END)
        for i, phrase in enumerate(self.phrases_list, 1):
            self.phrases_listbox.insert(tk.END, f"{i}. {phrase}")
        self.update_status(
            f"Phrase supprimée ({len(self.phrases_list)} restantes)")

    def clear_all_phrases(self):
        """Efface toutes les phrases."""
        if self.phrases_list:
            if messagebox.askyesno("Confirmation", "Effacer toutes les phrases?"):
                self.phrases_list.clear()
                self.phrases_listbox.delete(0, tk.END)
                self.multi_result_text.config(state='normal')
                self.multi_result_text.delete("1.0", tk.END)
                self.multi_result_text.config(state='disabled')
                self.update_status("Toutes les phrases effacées")

    def compare_multiple_phrases(self):
        """Compare toutes les phrases."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = self.calculator.compare_multiple_sentences(
                self.phrases_list)
            results.sort(key=lambda x: x[2], reverse=True)

            output = f"""
{'='*70}
COMPARAISON MULTIPLE DE PHRASES
{'='*70}

Nombre de phrases: {len(self.phrases_list)}
Nombre de comparaisons: {len(results)}

{'─'*70}
TOP 10 PAIRES LES PLUS SIMILAIRES:
{'─'*70}
"""
            for i, (idx1, idx2, sim) in enumerate(results[:10], 1):
                output += f"\n{i}. Similarité: {sim:.4f}\n"
                output += f"   Phrase {idx1+1}: {self.phrases_list[idx1][:60]}...\n"
                output += f"   Phrase {idx2+1}: {self.phrases_list[idx2][:60]}...\n"

            idx1, idx2, max_sim = self.calculator.get_most_similar_pair(
                self.phrases_list)
            output += f"\n{'─'*70}\n🏆 PAIRE LA PLUS SIMILAIRE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_sim:.4f}\n"

            idx1, idx2, max_dist = self.calculator.get_most_different_pair(
                self.phrases_list)
            output += f"\n📏 PAIRE LA PLUS DIFFÉRENTE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_dist:.4f} (distance)\n"
            output += f"{'='*70}\n"

            self.multi_result_text.config(state='normal')
            self.multi_result_text.delete("1.0", tk.END)
            self.multi_result_text.insert("1.0", output)
            self.multi_result_text.config(state='disabled')

            self.update_status(f"Comparaison: {len(results)} paires analysées")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_matrix_tab(self):
        """Onglet matrices."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Matrices  ")

        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        info_label = tk.Label(
            tab,
            text="Matrices de similarité et distance pour les phrases de 'Comparaison Multiple'.",
            font=('Arial', 10), wraplength=900
        )
        info_label.grid(row=0, column=0, pady=10, padx=20, sticky='w')

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="📊 Matrice Similarité",
            command=lambda: self.generate_matrix('similarity'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Matrice Distance",
            command=lambda: self.generate_matrix('distance'),
            bg=self.colors['primary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=1, padx=5, sticky='ew')

        matrix_frame = tk.LabelFrame(
            tab, text="🔢 Matrice",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        matrix_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        matrix_frame.rowconfigure(0, weight=1)
        matrix_frame.columnconfigure(0, weight=1)

        self.matrix_text = scrolledtext.ScrolledText(
            matrix_frame, font=('Courier', 9), wrap=tk.NONE, state='disabled')
        self.matrix_text.grid(row=0, column=0, sticky='nsew')

        xscrollbar = tk.Scrollbar(matrix_frame, orient='horizontal')
        xscrollbar.grid(row=1, column=0, sticky='ew')
        self.matrix_text.config(xscrollcommand=xscrollbar.set)
        xscrollbar.config(command=self.matrix_text.xview)

    def generate_matrix(self, matrix_type='similarity'):
        """Génère la matrice."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            if matrix_type == 'similarity':
                matrix = self.calculator.get_similarity_matrix(
                    self.phrases_list)
                title = "MATRICE DE SIMILARITÉ"
                legend = "Valeurs élevées = très similaires"
            else:
                matrix = self.calculator.get_distance_matrix(self.phrases_list)
                title = "MATRICE DE DISTANCE"
                legend = "Valeurs élevées = très différents"

            output = f"""
{'='*70}
{title}
{'='*70}

Phrases analysées:
"""
            for i, phrase in enumerate(self.phrases_list):
                output += f"  {i}: {phrase[:60]}...\n"

            output += f"\n{'─'*70}\nMatrice:\n\n     "
            for i in range(len(self.phrases_list)):
                output += f"{i:8}"
            output += "\n"

            for i, row in enumerate(matrix):
                output += f"{i:3}: "
                for value in row:
                    output += f"{value:8.4f}"
                output += "\n"

            output += f"\n{'='*70}\nLégende:\n"
            output += f"  • Diagonale = {'1.00' if matrix_type == 'similarity' else '0.00'}\n"
            output += f"  • {legend}\n"
            output += f"{'='*70}\n"

            self.matrix_text.config(state='normal')
            self.matrix_text.delete("1.0", tk.END)
            self.matrix_text.insert("1.0", output)
            self.matrix_text.config(state='disabled')

            self.update_status(f"Matrice {matrix_type} générée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_extreme_pairs_tab(self):
        """NOUVEAU: Onglet paires extrêmes."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Paires Extrêmes  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="ℹ️ Information",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Recherche automatique des paires les plus similaires et les plus différentes.",
            font=('Arial', 10), wraplength=900
        ).pack()

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="🏆 Paire la Plus Similaire",
            command=self.find_most_similar,
            bg=self.colors['success'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Paire la Plus Différente",
            command=self.find_most_different,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=1, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.extreme_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.extreme_result_text.pack(fill='both', expand=True)

    def find_most_similar(self):
        """Trouve la paire la plus similaire."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, similarity = self.calculator.get_most_similar_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_similarity(similarity)

            output = f"""
{'='*70}
🏆 PAIRE LA PLUS SIMILAIRE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📊 SCORE DE SIMILARITÉ: {similarity:.4f} ({similarity*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus similaire: {similarity:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def find_most_different(self):
        """Trouve la paire la plus différente."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, distance = self.calculator.get_most_different_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_distance(distance)

            output = f"""
{'='*70}
📏 PAIRE LA PLUS DIFFÉRENTE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📏 SCORE DE DISTANCE: {distance:.4f} ({distance*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus différente: {distance:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_demo_tests_tab(self):
        """NOUVEAU: Onglet tests automatiques."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Tests Auto  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="🧪 Tests Automatiques",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Lance des tests de démonstration comme jaccard_similarity.py",
            font=('Arial', 10)
        ).pack()

        tk.Button(
            tab, text="▶️ Lancer Tests de Démonstration",
            command=self.run_demo_tests,
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).grid(row=1, column=0, pady=20)

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats des Tests",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.demo_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.demo_result_text.pack(fill='both', expand=True)

    def run_demo_tests(self):
        """Lance les tests de démonstration."""
        examples = [
            ("Le chat mange des croquettes", "Le chien mange des croquettes"),
            ("Python est un langage de programmation",
             "Java est un langage de programmation"),
            ("Machine learning supervisé", "Apprentissage automatique supervisé"),
            ("Bonjour tout le monde", "Salut tout le monde"),
            ("Aucun mot en commun", "Différentes phrases complètement")
        ]

        output = f"""
{'='*70}
TESTS DE DÉMONSTRATION AUTOMATIQUES
{'='*70}

Configuration active:
  - Sensibilité à la casse: {'Activée' if self.case_sensitive.get() else 'Désactivée'}
  - Stop-words: {'Activés' if self.remove_stopwords.get() else 'Désactivés'}
  - Stemming: {'Activé' if self.use_stemming.get() else 'Désactivé'}

{'─'*70}
TESTS:
{'─'*70}
"""

        for i, (s1, s2) in enumerate(examples, 1):
            similarity = self.calculator.calculate_similarity(s1, s2)
            distance = self.calculator.calculate_distance(s1, s2)

            sim_interp = self.calculator.interpret_similarity(similarity)
            dist_interp = self.calculator.interpret_distance(distance)

            output += f"""
Test {i}:
  Phrase 1: '{s1}'
  Phrase 2: '{s2}'

  📊 SIMILARITÉ: {similarity:.4f}
     Catégorie: {sim_interp['emoji']} {sim_interp['category']}
     {sim_interp['general_interpretation'][:100]}...

  📏 DISTANCE: {distance:.4f}
     Catégorie: {dist_interp['emoji']} {dist_interp['category']}

  ✓ Vérification: {similarity:.4f} + {distance:.4f} = {similarity + distance:.4f}
{'─'*70}
"""

        output += f"\n{'='*70}\nTOUS LES TESTS TERMINÉS\n{'='*70}\n"

        self.demo_result_text.config(state='normal')
        self.demo_result_text.delete("1.0", tk.END)
        self.demo_result_text.insert("1.0", output)
        self.demo_result_text.config(state='disabled')

        self.update_status(f"{len(examples)} tests de démonstration terminés")

    def create_export_tab(self):
        """Onglet export."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Export  ")

        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame, text="Export des Résultats",
            font=('Arial', 16, 'bold')
        ).pack(pady=20)

        tk.Label(
            main_frame,
            text="Exportez les résultats au format CSV ou JSON",
            font=('Arial', 11), wraplength=600
        ).pack(pady=10)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=30)

        tk.Button(
            button_frame, text="💾 Exporter CSV",
            command=lambda: self.export_results('csv'),
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        tk.Button(
            button_frame, text="💾 Exporter JSON",
            command=lambda: self.export_results('json'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        self.export_status = tk.Label(main_frame, text="", font=('Arial', 10))
        self.export_status.pack(pady=20)

    def export_results(self, format_type):
        """Exporte les résultats."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = []
            for i in range(len(self.phrases_list)):
                for j in range(i + 1, len(self.phrases_list)):
                    detailed = self.calculator.calculate_distance_detailed(
                        self.phrases_list[i], self.phrases_list[j])
                    results.append(detailed)

            if format_type == 'csv':
                filename = self.calculator.export_results_to_csv(results)
            else:
                filename = self.calculator.export_results_to_json(results)

            if filename:
                self.export_status.config(
                    text=f"✓ Export réussi: {filename}",
                    fg=self.colors['success']
                )
                messagebox.showinfo("Succès", f"Fichier créé:\n{filename}")
            else:
                self.export_status.config(
                    text="❌ Échec de l'export",
                    fg=self.colors['danger']
                )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_about_tab(self):
        """Onglet À propos."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ℹ️ À Propos  ")

        main_frame = tk.Frame(tab, bg='white')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame,
            text="Calculateur de Similarité de Jaccard",
            font=('Arial', 16, 'bold'), bg='white', fg=self.colors['primary']
        ).pack(pady=10)

        description = """
✨ Version 2.2 - Interface Graphique Complète

Cette interface implémente TOUTES les fonctionnalités de jaccard_similarity.py:

• Calcul de similarité ET distance de Jaccard
• Stemming français amélioré
• Support des stop-words français (60+)
• Interprétation contextuelle (5 contextes)
• Export CSV et JSON
• Tests automatiques de démonstration
• Recherche de paires extrêmes
• Matrices complètes
• Analyse technique détaillée
• Interface flexible et responsive

📐 Formules:
Similarité(A,B) = |A ∩ B| / |A ∪ B|
Distance(A,B) = 1 - Similarité(A,B)
        """

        tk.Label(
            main_frame, text=description, font=('Arial', 10),
            bg='white', justify='left'
        ).pack(pady=20)

        team_frame = tk.LabelFrame(
            main_frame, text="👥 Équipe",
            font=('Arial', 11, 'bold'), bg='white', padx=20, pady=15
        )
        team_frame.pack(fill='x', pady=10)

        for member in ["OUEDRAOGO Lassina", "OUEDRAOGO Rasmane", "POUBERE Abdourazakou"]:
            tk.Label(
                team_frame, text=f"• {member}",
                font=('Arial', 10), bg='white'
            ).pack(anchor='w', pady=2)

        tk.Label(
            main_frame,
            text="📚 Machine Learning non Supervisé\n🎓 Octobre 2025 - Version Complète v2.2",
            font=('Arial', 10), bg='white', fg=self.colors['dark']
        ).pack(pady=20)

    def create_status_bar(self):
        """Crée la barre de statut."""
        self.status_bar = tk.Label(
            self.root,
            text="Prêt | Interface Complète avec TOUTES les fonctionnalités",
            bd=1, relief=tk.SUNKEN, anchor='w', font=('Arial', 9)
        )
        self.status_bar.grid(row=3, column=0, sticky='ew')

    def update_status(self, message):
        """Met à jour le statut."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main():
    """Point d'entrée."""
    root = tk.Tk()
    app = JaccardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

# run.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement rapide pour le Projet Jaccard
Machine Learning non Supervisé - Similarité de Jaccard
Version 2.1 - Amélioration du mode interactif avec configuration
Usage: python run.py
"""

import os
import sys
import subprocess
from pathlib import Path
import platform

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.1"
DATE = "Novembre 2025"
AUTEURS = "OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou"

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def print_banner():
    """Affiche la bannière du projet"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     📊  CALCULATEUR DE SIMILARITÉ DE JACCARD - v2.1               ║
║          Machine Learning non Supervisé                           ║
║                                                                    ║
║     👥  OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou║
║     📅  Novembre 2025                                              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)


def clear_screen():
    """Efface l'écran selon l'OS"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_separator(char="=", length=70):
    """Affiche une ligne de séparation"""
    print(char * length)


def get_python_command():
    """Retourne la commande Python appropriée selon l'OS"""
    if platform.system() == "Windows":
        return "python"
    else:
        return "python3"


# ============================================================================
# VÉRIFICATIONS
# ============================================================================

def check_environment():
    """Vérifie que l'environnement est prêt"""
    issues = []
    warnings = []

    # Vérifier la version de Python
    if sys.version_info < (3, 6):
        issues.append("❌ Python 3.6+ requis")
    elif sys.version_info >= (3, 6):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Vérifier la présence des fichiers essentiels
    required_files = {
        "jaccard_similarity.py": "Script principal",
        "jaccard_gui.py": "Interface graphique",
        "test_jaccard.py": "Tests unitaires",
    }

    print("\n🔍 Vérification des fichiers...")
    for file, description in required_files.items():
        if Path(file).exists():
            print(f"  ✅ {file:25s} - {description}")
        else:
            issues.append(f"❌ Fichier manquant: {file} ({description})")
            print(f"  ❌ {file:25s} - MANQUANT")

    # Vérifier les fichiers optionnels
    optional_files = {
        "examples/demo.py": "Démonstrations",
        "README.md": "Documentation",
        "QUICKSTART.md": "Guide rapide",
    }

    print("\n📋 Fichiers optionnels:")
    for file, description in optional_files.items():
        if Path(file).exists():
            print(f"  ✅ {file:25s} - {description}")
        else:
            warnings.append(f"⚠️  Fichier optionnel absent: {file}")
            print(f"  ⚠️  {file:25s} - Absent (optionnel)")

    # Créer le dossier examples s'il n'existe pas
    if not Path("examples").exists():
        print("\n📁 Création du dossier examples/...")
        Path("examples").mkdir(exist_ok=True)

    return issues, warnings


def check_dependencies():
    """Vérifie que les dépendances sont installées"""
    print("\n🔍 Vérification des dépendances...")

    all_ok = True

    # Dépendances essentielles (bibliothèque standard)
    essential_modules = [
        ("re", "Expressions régulières"),
        ("json", "Traitement JSON"),
        ("csv", "Traitement CSV"),
        ("argparse", "Arguments CLI"),
    ]

    for module, description in essential_modules:
        try:
            __import__(module)
            print(f"  ✅ {module:20s} - {description}")
        except ImportError:
            print(f"  ❌ {module:20s} - MANQUANT")
            all_ok = False

    # Dépendances optionnelles pour l'interface graphique
    print("\n📦 Dépendances optionnelles (GUI):")
    try:
        import tkinter
        print(f"  ✅ tkinter              - Interface graphique")
    except ImportError:
        print(f"  ⚠️  tkinter              - Absent (GUI non disponible)")

    return all_ok


# ============================================================================
# FONCTIONS DE CONFIGURATION
# ============================================================================

def configure_interactive_mode():
    """
    Configure les options pour le mode interactif.
    Retourne les arguments de configuration.
    """
    clear_screen()
    print_banner()
    print("\n" + "="*70)
    print("          ⚙️  CONFIGURATION DU MODE INTERACTIF")
    print("="*70)
    
    print("\nAvant de commencer, choisissez vos options de traitement:")
    print()
    
    # Option 1: Sensibilité à la casse
    print("─" * 70)
    print("1️⃣  SENSIBILITÉ À LA CASSE")
    print("─" * 70)
    print("   • OUI: 'Python' et 'python' sont considérés comme différents")
    print("   • NON: 'Python' et 'python' sont considérés comme identiques")
    print()
    case_sensitive = input("   👉 Activer la sensibilité à la casse? (o/N): ").lower().strip() == 'o'
    
    # Option 2: Ponctuation
    print("\n" + "─" * 70)
    print("2️⃣  PONCTUATION")
    print("─" * 70)
    print("   • SUPPRIMER: 'Hello!' devient 'Hello' (recommandé)")
    print("   • CONSERVER: 'Hello!' reste 'Hello!'")
    print()
    keep_punct = input("   👉 Conserver la ponctuation? (o/N): ").lower().strip() == 'o'
    remove_punctuation = not keep_punct
    
    # Option 3: Stop-words
    print("\n" + "─" * 70)
    print("3️⃣  STOP-WORDS FRANÇAIS")
    print("─" * 70)
    print("   Les stop-words sont des mots très courants comme:")
    print("   'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'dans'...")
    print()
    print("   • OUI: Ces mots seront ignorés (meilleure précision)")
    print("   • NON: Tous les mots sont pris en compte")
    print()
    remove_stopwords = input("   👉 Retirer les stop-words français? (o/N): ").lower().strip() == 'o'
    
    # Option 4: Stemming
    print("\n" + "─" * 70)
    print("4️⃣  STEMMING FRANÇAIS")
    print("─" * 70)
    print("   Le stemming réduit les mots à leur racine:")
    print("   'manger', 'mange', 'mangé' → 'mang'")
    print("   'programmation', 'programmer' → 'programm'")
    print()
    print("   • OUI: Détecte mieux les similarités (recommandé)")
    print("   • NON: Les mots doivent être identiques")
    print()
    use_stemming = input("   👉 Activer le stemming français? (o/N): ").lower().strip() == 'o'
    
    # Résumé de la configuration
    print("\n" + "="*70)
    print("          📋 RÉSUMÉ DE VOTRE CONFIGURATION")
    print("="*70)
    print(f"  • Sensibilité à la casse:  {'✅ OUI' if case_sensitive else '❌ NON'}")
    print(f"  • Suppression ponctuation: {'✅ OUI' if remove_punctuation else '❌ NON'}")
    print(f"  • Retrait stop-words:      {'✅ OUI' if remove_stopwords else '❌ NON'}")
    print(f"  • Stemming français:       {'✅ OUI' if use_stemming else '❌ NON'}")
    print("="*70)
    
    input("\n👉 Appuyez sur Entrée pour lancer le mode interactif...")
    
    # Construire les arguments
    args = ["--interactive"]
    if case_sensitive:
        args.append("--case-sensitive")
    if not remove_punctuation:
        args.append("--keep-punctuation")
    if remove_stopwords:
        args.append("--remove-stopwords")
    if use_stemming:
        args.append("--use-stemming")
    
    return args


# ============================================================================
# FONCTIONS DE LANCEMENT
# ============================================================================

def launch_default_tests():
    """Lance les tests automatiques par défaut"""
    print("\n🧪 Lancement des tests automatiques...")
    print_separator()
    print("Ceci lance les 5 exemples de démonstration prédéfinis")
    print("avec analyse de similarité ET distance.\n")

    python_cmd = get_python_command()

    try:
        subprocess.run([python_cmd, "jaccard_similarity.py"])
        return True
    except FileNotFoundError:
        print(f"❌ Erreur: jaccard_similarity.py introuvable")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def launch_interactive_mode():
    """Lance le mode interactif avec configuration préalable"""
    print("\n💬 Configuration du mode interactif...")
    print_separator()
    
    # Demander la configuration
    args = configure_interactive_mode()
    
    print("\n💬 Lancement du mode interactif...")
    print_separator()
    print("Vous pourrez saisir vos propres phrases pour les comparer.")
    print("Tapez 'quit' pour quitter le mode interactif.\n")

    python_cmd = get_python_command()

    try:
        subprocess.run([python_cmd, "jaccard_similarity.py"] + args)
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def launch_gui():
    """Lance l'interface graphique"""
    print("\n🖥️  Lancement de l'interface graphique...")
    print_separator()
    print("L'interface Tkinter va s'ouvrir dans une nouvelle fenêtre.")
    print("Pour fermer: utilisez le bouton de fermeture de la fenêtre.\n")

    python_cmd = get_python_command()

    # Vérifier que tkinter est disponible
    try:
        import tkinter
    except ImportError:
        print("❌ Tkinter n'est pas installé.")
        print("💡 Installez tkinter selon votre système:")
        print("   - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   - Fedora: sudo dnf install python3-tkinter")
        print("   - macOS: Tkinter est inclus avec Python")
        print("   - Windows: Tkinter est inclus avec Python")
        return False

    try:
        subprocess.run([python_cmd, "jaccard_gui.py"])
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def launch_unit_tests():
    """Lance les tests unitaires"""
    print("\n🧪 Lancement des tests unitaires...")
    print_separator()
    print("Exécution de 50+ tests couvrant toutes les fonctionnalités.\n")

    python_cmd = get_python_command()

    try:
        subprocess.run([python_cmd, "test_jaccard.py"])
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def launch_demos():
    """Lance les démonstrations avancées"""
    print("\n🎯 Lancement des démonstrations avancées...")
    print_separator()
    print("11 démonstrations interactives de cas d'usage pratiques.\n")

    python_cmd = get_python_command()

    if not Path("examples/demo.py").exists():
        print("❌ Fichier examples/demo.py introuvable")
        print("💡 Assurez-vous que le dossier examples/ contient demo.py")
        return False

    try:
        subprocess.run([python_cmd, "examples/demo.py"])
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def show_advanced_options():
    """Affiche et gère les options avancées"""
    print("\n⚙️  OPTIONS AVANCÉES")
    print_separator()
    print("\n📋 Configuration disponible:")
    print()
    print("1. Tests avec stop-words (retire 'le', 'la', 'les', etc.)")
    print("   Commande: python jaccard_similarity.py --remove-stopwords")
    print()
    print("2. Tests avec stemming (manger → mang)")
    print("   Commande: python jaccard_similarity.py --use-stemming")
    print()
    print("3. Configuration optimale (stop-words + stemming)")
    print("   Commande: python jaccard_similarity.py --remove-stopwords --use-stemming")
    print()
    print("4. Export des résultats en CSV")
    print("   Commande: python jaccard_similarity.py --export csv")
    print()
    print("5. Export des résultats en JSON")
    print("   Commande: python jaccard_similarity.py --export json")
    print()
    print("6. Mode sensible à la casse")
    print("   Commande: python jaccard_similarity.py --case-sensitive")
    print()
    print("7. Conserver la ponctuation")
    print("   Commande: python jaccard_similarity.py --keep-punctuation")
    print()
    print_separator()
    print("\n💡 Vous pouvez combiner plusieurs options:")
    print("   python jaccard_similarity.py --interactive --remove-stopwords --use-stemming")
    print()
    print("📚 Consultez README.md ou HARMONISATION.md pour plus de détails")
    print()

    input("\n👉 Appuyez sur Entrée pour continuer...")


def show_help():
    """Affiche l'aide complète"""
    print("\n📖 AIDE ET DOCUMENTATION")
    print_separator()
    print()
    print("🎯 MODES D'EXÉCUTION:")
    print("  1. Tests automatiques    - Exemples prédéfinis avec analyse")
    print("  2. Mode interactif       - Saisissez vos propres phrases")
    print("  3. Interface graphique   - Interface Tkinter complète")
    print("  4. Tests unitaires       - Validation du code (50+ tests)")
    print("  5. Démonstrations        - 11 cas d'usage pratiques")
    print()
    print("⚙️  OPTIONS DE CONFIGURATION:")
    print("  --case-sensitive         - Respecte la casse (Python ≠ python)")
    print("  --keep-punctuation       - Garde la ponctuation")
    print("  --remove-stopwords       - Retire les stop-words français")
    print("  --use-stemming          - Applique le stemming français")
    print("  --export csv|json|both  - Exporte les résultats")
    print()
    print("📚 DOCUMENTATION:")
    print("  README.md               - Documentation complète du projet")
    print("  QUICKSTART.md           - Guide de démarrage rapide")
    print("  HARMONISATION.md        - Guide d'harmonisation v2.0")
    print()
    print("💡 EXEMPLES D'UTILISATION:")
    print("  python jaccard_similarity.py")
    print("  python jaccard_similarity.py --interactive")
    print("  python jaccard_similarity.py --remove-stopwords --use-stemming")
    print("  python jaccard_similarity.py --export json")
    print()
    print_separator()

    input("\n👉 Appuyez sur Entrée pour continuer...")


# ============================================================================
# MENU PRINCIPAL
# ============================================================================

def show_main_menu():
    """Affiche le menu principal"""
    print("\n" + "="*70)
    print("                    📋 MENU PRINCIPAL")
    print("="*70)
    print()
    print("  1. 🧪 Tests automatiques (par défaut)")
    print("       └─ Lance 5 exemples de démonstration")
    print()
    print("  2. 💬 Mode interactif (avec configuration)")
    print("       └─ Configurez vos options puis saisissez vos phrases")
    print()
    print("  3. 🖥️  Interface graphique")
    print("       └─ Lance l'interface Tkinter complète")
    print()
    print("  4. 🧪 Tests unitaires")
    print("       └─ Exécute les 50+ tests de validation")
    print()
    print("  5. 🎯 Démonstrations avancées")
    print("       └─ 11 cas d'usage pratiques (plagiat, clustering, etc.)")
    print()
    print("  6. ⚙️  Options avancées")
    print("       └─ Configuration et personnalisation")
    print()
    print("  7. 📖 Aide et documentation")
    print("       └─ Guide complet d'utilisation")
    print()
    print("  8. 🚪 Quitter")
    print()
    print("="*70)


def get_user_choice():
    """Demande le choix de l'utilisateur"""
    while True:
        try:
            choice = input("\n👉 Votre choix (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return choice
            else:
                print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 8.")
        except KeyboardInterrupt:
            print("\n\n👋 Programme interrompu par l'utilisateur")
            sys.exit(0)
        except Exception:
            print("❌ Erreur de saisie. Veuillez réessayer.")


def handle_choice(choice):
    """Gère le choix de l'utilisateur"""
    if choice == '1':
        return launch_default_tests()
    elif choice == '2':
        return launch_interactive_mode()
    elif choice == '3':
        return launch_gui()
    elif choice == '4':
        return launch_unit_tests()
    elif choice == '5':
        return launch_demos()
    elif choice == '6':
        show_advanced_options()
        return True
    elif choice == '7':
        show_help()
        return True
    elif choice == '8':
        return False
    else:
        print("❌ Choix invalide")
        return True


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale"""
    try:
        # Affichage de la bannière
        clear_screen()
        print_banner()

        # Vérifications initiales
        print("\n🔍 Vérification de l'environnement...")
        print_separator()

        issues, warnings = check_environment()

        # Afficher les avertissements
        if warnings:
            print("\n⚠️  Avertissements:")
            for warning in warnings:
                print(f"  {warning}")

        # Vérifier s'il y a des problèmes bloquants
        if issues:
            print("\n❌ PROBLÈMES DÉTECTÉS:")
            for issue in issues:
                print(f"  {issue}")
            print("\n💡 Veuillez résoudre ces problèmes avant de continuer.")
            print("   Consultez README.md pour l'installation.")
            input("\n👉 Appuyez sur Entrée pour quitter...")
            sys.exit(1)

        # Vérification des dépendances
        if not check_dependencies():
            print("\n❌ Certaines dépendances sont manquantes.")
            print("💡 Toutes les dépendances devraient être disponibles dans Python standard.")
            response = input("\n❓ Continuer quand même? (o/n): ").lower()
            if response != 'o':
                sys.exit(1)

        print("\n✅ Environnement prêt!")
        input("\n👉 Appuyez sur Entrée pour continuer...")

        # Boucle principale du menu
        while True:
            clear_screen()
            print_banner()
            show_main_menu()

            choice = get_user_choice()

            if choice == '8':
                clear_screen()
                print("\n" + "="*70)
                print("          👋 MERCI D'AVOIR UTILISÉ LE CALCULATEUR JACCARD")
                print("="*70)
                print()
                print(f"  📊 Version {VERSION}")
                print(f"  👥 {AUTEURS}")
                print(f"  📅 {DATE}")
                print()
                print("  🎓 Projet de Machine Learning non Supervisé")
                print("  🔗 Consultez notre documentation pour plus d'informations")
                print()
                print("="*70)
                print()
                break

            # Exécuter le choix
            print("\n")
            success = handle_choice(choice)

            # Demander si l'utilisateur veut continuer
            if success:
                print("\n" + "="*70)
                response = input("\n🔄 Retour au menu principal? (o/n): ").lower()
                if response != 'o':
                    print("\n👋 À bientôt!")
                    break
            else:
                # En cas d'erreur, proposer de réessayer
                print("\n" + "="*70)
                response = input("\n🔄 Retour au menu principal? (o/n): ").lower()
                if response != 'o':
                    break

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("✅ Programme interrompu par l'utilisateur")
        print("="*70)
        print("\n👋 À bientôt!\n")
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERREUR INATTENDUE")
        print("="*70)
        print(f"\n{e}")
        print("\n💡 Consultez README.md ou contactez l'équipe du projet")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()
```

# examples/demo.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démonstration avancée pour le calculateur de similarité de Jaccard
Montre différents cas d'usage pratiques et applications réelles

Usage: python examples/demo.py
"""

import sys
import os
import time

# Ajout du répertoire parent au path pour pouvoir importer le module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from jaccard_similarity import JaccardSimilarity
except ImportError:
    print("Erreur : Impossible d'importer jaccard_similarity")
    print(f"Répertoire actuel : {current_dir}")
    print(f"Répertoire parent : {parent_dir}")
    print(f"Chemin Python : {sys.path[:3]}")
    print("\nVérifiez que jaccard_similarity.py est bien dans le répertoire parent.")
    sys.exit(1)


def demo_basic_usage():
    """Quelques exemples d'utilisation simple."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 1: Utilisation de base")
    print("="*70)

    calculator = JaccardSimilarity()

    examples = [
        ("Le chat mange des croquettes", "Le chien mange des croquettes"),
        ("Bonjour tout le monde", "Salut tout le monde"),
        ("Python est fantastique", "Java est fantastique"),
        ("Machine Learning", "Apprentissage automatique"),
        ("Phrase identique", "Phrase identique")
    ]

    for i, (phrase1, phrase2) in enumerate(examples, 1):
        similarity = calculator.calculate_similarity(phrase1, phrase2)

        print(f"\nExemple {i}:")
        print(f"  Phrase 1: '{phrase1}'")
        print(f"  Phrase 2: '{phrase2}'")
        print(f"  Similarité: {similarity:.4f}")

        # Interprétation du score
        if similarity >= 0.8:
            interpretation = "Très similaires"
        elif similarity >= 0.5:
            interpretation = "Moyennement similaires"
        elif similarity > 0:
            interpretation = "Peu similaires"
        else:
            interpretation = "Pas de similarité"

        print(f"  Interprétation: {interpretation}")


def demo_configuration_options():
    """Test des options disponibles (casse, ponctuation)."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 2: Options de configuration")
    print("="*70)

    phrase1 = "Bonjour, Comment allez-vous ?"
    phrase2 = "BONJOUR comment allez vous"

    configs = [
        (JaccardSimilarity(), "Configuration par défaut"),
        (JaccardSimilarity(case_sensitive=True), "Sensible à la casse"),
        (JaccardSimilarity(remove_punctuation=False), "Avec ponctuation"),
        (JaccardSimilarity(case_sensitive=True, remove_punctuation=False),
         "Casse + ponctuation")
    ]

    print(f"\nPhrase 1: '{phrase1}'")
    print(f"Phrase 2: '{phrase2}'")
    print()

    for calc, description in configs:
        sim = calc.calculate_similarity(phrase1, phrase2)
        print(f"  {description:30s}: {sim:.4f}")

    print("\nObservation: Les options de configuration changent")
    print("             significativement les résultats")


def demo_plagiarism_detection():
    """Exemple d'application pour détecter le plagiat."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 3: Détection de plagiat")
    print("="*70)

    calculator = JaccardSimilarity()

    original = ("L'intelligence artificielle transforme notre société "
                "en automatisant les tâches complexes")

    documents = [
        ("Copie exacte",
         "L'intelligence artificielle transforme notre société en automatisant les tâches complexes"),
        ("Mots réarrangés",
         "Notre société transforme l'intelligence artificielle en automatisant les tâches complexes"),
        ("Synonymes partiels",
         "L'IA transforme notre société en automatisant les tâches difficiles"),
        ("Paraphrase",
         "L'automatisation des processus complexes change notre monde grâce à l'IA"),
        ("Texte différent",
         "Les océans contiennent une biodiversité marine extraordinaire et fragile")
    ]

    print(f"\nDocument original:\n  '{original}'\n")
    print("Analyse de similarité:\n")

    for nom, doc in documents:
        similarity = calculator.calculate_similarity(original, doc)

        # Détermination du niveau de suspicion
        if similarity >= 0.8:
            niveau = "PLAGIAT PROBABLE"
        elif similarity >= 0.5:
            niveau = "SUSPICION ÉLEVÉE"
        elif similarity >= 0.2:
            niveau = "SUSPICION MODÉRÉE"
        else:
            niveau = "ORIGINAL"

        print(f"{nom}:")
        print(f"  '{doc}'")
        print(f"  Similarité: {similarity:.4f} - {niveau}")
        print()


def demo_document_clustering():
    """Regroupement de documents par similarité."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 4: Regroupement de documents (clustering)")
    print("="*70)

    calculator = JaccardSimilarity()

    documents = [
        "Python est un langage de programmation polyvalent et puissant",
        "Java est un langage orienté objet très populaire en entreprise",
        "Le machine learning utilise des algorithmes pour analyser les données",
        "L'intelligence artificielle révolutionne de nombreux secteurs",
        "JavaScript permet de créer des sites web interactifs et dynamiques",
        "Les réseaux de neurones simulent le fonctionnement du cerveau humain",
        "C++ est un langage performant pour le développement système",
        "L'apprentissage automatique nécessite beaucoup de données d'entraînement"
    ]

    print("\nCollection de documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    # Comparaison de tous les documents
    results = calculator.compare_multiple_sentences(documents)
    results.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 des paires les plus similaires:")
    for i, (idx1, idx2, sim) in enumerate(results[:5], 1):
        print(f"\n  {i}. Documents {idx1+1} et {idx2+1} (similarité: {sim:.4f}):")
        print(f"     • {documents[idx1][:60]}...")
        print(f"     • {documents[idx2][:60]}...")

    # Identification des clusters
    print("\nClusters potentiels (similarité > 0.3):")
    clusters = [(idx1, idx2, sim) for idx1, idx2, sim in results if sim > 0.3]

    if clusters:
        for i, (idx1, idx2, sim) in enumerate(clusters, 1):
            print(f"  Cluster {i}: Documents {idx1+1} et {idx2+1} ({sim:.4f})")
    else:
        print("  Aucun cluster détecté avec ce seuil")


def demo_search_engine():
    """Moteur de recherche simple basé sur Jaccard."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 5: Moteur de recherche simple")
    print("="*70)

    calculator = JaccardSimilarity()

    articles = [
        "Les voitures électriques révolutionnent le transport urbain",
        "L'énergie solaire devient de plus en plus accessible",
        "Les smartphones modernes intègrent l'intelligence artificielle",
        "La cuisine française est reconnue mondialement",
        "Les véhicules autonomes transforment la mobilité urbaine",
        "L'énergie renouvelable réduit l'empreinte carbone",
        "L'IA améliore les performances des téléphones portables",
        "La gastronomie italienne influence la cuisine mondiale",
        "Les transports en commun électriques se développent",
        "Les panneaux photovoltaïques équipent de plus en plus de maisons"
    ]

    print("\nBase d'articles:")
    for i, article in enumerate(articles, 1):
        print(f"  {i:2d}. {article}")

    queries = [
        "voiture électrique transport",
        "énergie solaire maison",
        "intelligence artificielle téléphone",
        "cuisine gastronomie"
    ]

    for query in queries:
        print(f"\nRecherche: '{query}'")
        print("  Résultats (score de pertinence):")

        # Calcul des scores pour tous les articles
        scores = [(i, calculator.calculate_similarity(query, article))
                  for i, article in enumerate(articles)]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Affichage des 3 meilleurs résultats
        found_results = False
        for rank, (idx, score) in enumerate(scores[:3], 1):
            if score > 0:
                print(f"    {rank}. (Score: {score:.3f}) {articles[idx]}")
                found_results = True

        if not found_results:
            print("    Aucun résultat pertinent trouvé")


def demo_performance_analysis():
    """Test de performance avec différentes tailles de données."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 6: Analyse de performance")
    print("="*70)

    calculator = JaccardSimilarity()

    sizes = [10, 50, 100, 200]

    print("\nTests de performance avec différentes tailles:\n")

    for size in sizes:
        # Génération de phrases de test
        sentences = [f"phrase de test numéro {i} avec quelques mots"
                     for i in range(size)]

        # Mesure du temps d'exécution
        start_time = time.time()
        results = calculator.compare_multiple_sentences(sentences)
        end_time = time.time()

        execution_time = end_time - start_time
        comparisons = len(results)
        comp_per_sec = comparisons / \
            execution_time if execution_time > 0 else float('inf')

        print(f"  {size:3d} phrases → {comparisons:5d} comparaisons "
              f"en {execution_time:.3f}s ({comp_per_sec:.0f} comp/s)")

    print("\nConclusion: L'algorithme reste rapide même avec beaucoup de phrases")


def demo_similarity_matrix():
    """Construction d'une matrice de similarité."""
    print("\n" + "="*70)
    print("DÉMONSTRATION 7: Matrice de similarité")
    print("="*70)

    calculator = JaccardSimilarity()

    sentences = [
        "Le chat noir dort",
        "Le chien blanc court",
        "Le chat blanc mange",
        "Un oiseau vole haut"
    ]

    print("\nPhrases analysées:")
    for i, s in enumerate(sentences):
        print(f"  {i}: {s}")

    matrix = calculator.get_similarity_matrix(sentences)

    # Affichage de la matrice
    print("\nMatrice de similarité:")
    print("       ", end="")
    for i in range(len(sentences)):
        print(f"   {i}  ", end="")
    print()

    for i, row in enumerate(matrix):
        print(f"  {i}  ", end="")
        for sim in row:
            print(f" {sim:.2f} ", end="")
        print()

    print("\nInterprétation:")
    print("  - Diagonale = 1.00 (phrase identique à elle-même)")
    print("  - Valeurs élevées = phrases très similaires")
    print("  - Valeurs faibles = phrases peu similaires")


def main():
    """Point d'entrée du programme."""
    print("="*70)
    print("DÉMONSTRATIONS PRATIQUES - SIMILARITÉ DE JACCARD")
    print("="*70)
    print("\nCe script présente différentes applications pratiques")
    print("du calculateur de similarité de Jaccard.")

    try:
        demo_basic_usage()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_configuration_options()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_plagiarism_detection()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_document_clustering()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_search_engine()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_performance_analysis()
        input("\nAppuyez sur Entrée pour continuer...")

        demo_similarity_matrix()

        print("\n" + "="*70)
        print("FIN DES DÉMONSTRATIONS")
        print("="*70)
        print("\nVous pouvez maintenant utiliser jaccard_similarity.py")
        print("pour vos propres projets de machine learning.")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nDémonstration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\nErreur pendant la démonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

# test_jaccard.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le calculateur de similarité de Jaccard
Projet de Machine Learning non Supervisé

Usage: python test_jaccard.py
"""

import unittest
import time
import sys
import os

from jaccard_similarity import JaccardSimilarity


class TestJaccardSimilarityBasic(unittest.TestCase):
    """Classe pour tester les fonctionnalités de base."""

    def setUp(self):
        """Initialisation des calculateurs pour les tests."""
        self.calculator = JaccardSimilarity()
        self.calculator_case_sensitive = JaccardSimilarity(case_sensitive=True)
        self.calculator_with_punct = JaccardSimilarity(
            remove_punctuation=False)

    def test_identical_sentences(self):
        """Vérification qu'une phrase identique à elle-même donne 1.0."""
        sentence = "Le chat mange des croquettes"
        similarity = self.calculator.calculate_similarity(sentence, sentence)
        self.assertEqual(similarity, 1.0)

    def test_completely_different_sentences(self):
        """Deux phrases complètement différentes doivent donner 0.0."""
        sentence1 = "Le chat mange"
        sentence2 = "Python programmation"
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        self.assertEqual(similarity, 0.0)

    def test_partial_similarity(self):
        """Test du calcul avec des phrases qui ont des mots en commun."""
        sentence1 = "Le chat mange des croquettes"
        sentence2 = "Le chien mange des croquettes"

        # 4 mots en commun (le, mange, des, croquettes)
        # 6 mots au total (le, chat, chien, mange, des, croquettes)
        # Donc 4/6 = 0.6667
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        self.assertAlmostEqual(similarity, 4/6, places=4)

    def test_empty_sentences(self):
        """Test du comportement avec des chaînes vides."""
        similarity_both_empty = self.calculator.calculate_similarity("", "")
        self.assertEqual(similarity_both_empty, 0.0)

        similarity_one_empty = self.calculator.calculate_similarity(
            "", "hello world")
        self.assertEqual(similarity_one_empty, 0.0)

    def test_single_word_sentences(self):
        """Comparaison de phrases d'un seul mot."""
        similarity_same = self.calculator.calculate_similarity("chat", "chat")
        self.assertEqual(similarity_same, 1.0)

        similarity_diff = self.calculator.calculate_similarity("chat", "chien")
        self.assertEqual(similarity_diff, 0.0)


class TestPreprocessing(unittest.TestCase):
    """Tests du prétraitement des phrases."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_preprocess_basic(self):
        """Test simple du prétraitement."""
        result = self.calculator.preprocess_sentence("Hello World")
        expected = {'hello', 'world'}
        self.assertEqual(result, expected)

    def test_preprocess_punctuation(self):
        """La ponctuation doit être supprimée."""
        result = self.calculator.preprocess_sentence("Hello, World!")
        expected = {'hello', 'world'}
        self.assertEqual(result, expected)

    def test_preprocess_empty(self):
        """Une phrase vide doit retourner un ensemble vide."""
        result = self.calculator.preprocess_sentence("")
        expected = set()
        self.assertEqual(result, expected)

    def test_preprocess_accents(self):
        """Les accents français doivent être préservés."""
        result = self.calculator.preprocess_sentence("Café français")
        expected = {'café', 'français'}
        self.assertEqual(result, expected)

    def test_preprocess_multiple_spaces(self):
        """Les espaces multiples doivent être gérés correctement."""
        result = self.calculator.preprocess_sentence("Le  chat   mange")
        expected = {'le', 'chat', 'mange'}
        self.assertEqual(result, expected)

    def test_preprocess_spaces_only(self):
        """Une phrase avec que des espaces doit donner un ensemble vide."""
        result = self.calculator.preprocess_sentence("   ")
        self.assertEqual(result, set())


class TestCaseAndPunctuation(unittest.TestCase):
    """Tests des options de casse et ponctuation."""

    def test_case_sensitivity_off(self):
        """Par défaut, la casse ne devrait pas être prise en compte."""
        calculator = JaccardSimilarity(case_sensitive=False)
        similarity = calculator.calculate_similarity(
            "Hello World", "hello world")
        self.assertEqual(similarity, 1.0)

    def test_case_sensitivity_on(self):
        """Quand case_sensitive=True, la casse doit être respectée."""
        calculator = JaccardSimilarity(case_sensitive=True)
        sentence1 = "Hello World"
        sentence2 = "hello world"

        similarity = calculator.calculate_similarity(sentence1, sentence2)
        self.assertLess(similarity, 1.0)

    def test_punctuation_removal(self):
        """La ponctuation est supprimée par défaut."""
        calculator = JaccardSimilarity(remove_punctuation=True)
        similarity = calculator.calculate_similarity(
            "Hello, world!", "Hello world")
        self.assertEqual(similarity, 1.0)

    def test_punctuation_kept(self):
        """Avec remove_punctuation=False, la ponctuation est gardée."""
        calculator = JaccardSimilarity(remove_punctuation=False)
        similarity = calculator.calculate_similarity("Hello!", "Hello")
        self.assertLess(similarity, 1.0)


class TestDetailedCalculation(unittest.TestCase):
    """Tests pour le calcul détaillé."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_detailed_result_structure(self):
        """Vérification que le résultat détaillé contient toutes les informations."""
        result = self.calculator.calculate_similarity_detailed(
            "hello world", "hello python"
        )

        # On vérifie que toutes les clés sont présentes
        required_keys = [
            'sentence1', 'sentence2', 'words_set1', 'words_set2',
            'intersection', 'union', 'intersection_size', 'union_size',
            'jaccard_similarity'
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_detailed_calculation_values(self):
        """Test des valeurs retournées par le calcul détaillé."""
        result = self.calculator.calculate_similarity_detailed(
            "hello world", "hello python"
        )

        self.assertEqual(result['words_set1'], {'hello', 'world'})
        self.assertEqual(result['words_set2'], {'hello', 'python'})
        self.assertEqual(result['intersection'], {'hello'})
        self.assertEqual(result['union'], {'hello', 'world', 'python'})
        self.assertEqual(result['intersection_size'], 1)
        self.assertEqual(result['union_size'], 3)
        self.assertAlmostEqual(result['jaccard_similarity'], 1/3, places=3)


class TestMultipleComparisons(unittest.TestCase):
    """Tests pour comparer plusieurs phrases à la fois."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_compare_multiple_sentences(self):
        """Test de la comparaison de plusieurs phrases."""
        sentences = [
            "Le chat mange",
            "Le chien mange",
            "Python programmation"
        ]

        results = self.calculator.compare_multiple_sentences(sentences)

        # Avec 3 phrases, on devrait avoir 3 comparaisons: (0,1), (0,2), (1,2)
        self.assertEqual(len(results), 3)

        # Chaque résultat doit être bien formaté
        for idx1, idx2, similarity in results:
            self.assertIsInstance(idx1, int)
            self.assertIsInstance(idx2, int)
            self.assertIsInstance(similarity, float)
            self.assertTrue(0 <= similarity <= 1)
            self.assertLess(idx1, idx2)

    def test_get_most_similar_pair(self):
        """Recherche de la paire la plus similaire dans une liste."""
        sentences = [
            "Le chat mange des croquettes",
            "Python est génial",
            "Le chien mange des croquettes",
            "Java est bien"
        ]

        idx1, idx2, max_similarity = self.calculator.get_most_similar_pair(
            sentences)

        # Les phrases 0 et 2 devraient être les plus similaires
        self.assertTrue((idx1 == 0 and idx2 == 2) or (idx1 == 2 and idx2 == 0))
        self.assertGreater(max_similarity, 0.5)

    def test_similarity_matrix(self):
        """Test de la génération d'une matrice de similarité."""
        sentences = ["chat", "chien", "oiseau"]
        matrix = self.calculator.get_similarity_matrix(sentences)

        # La matrice doit être 3x3
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)

        # La diagonale doit contenir des 1.0
        for i in range(3):
            self.assertEqual(matrix[i][i], 1.0)

        # La matrice doit être symétrique
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(matrix[i][j], matrix[j][i], places=10)


class TestRealWorldExamples(unittest.TestCase):
    """Tests avec des cas réalistes."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_similar_news_articles(self):
        """Test avec des titres d'actualité similaires."""
        news1 = "Le président annonce de nouvelles mesures économiques"
        news2 = "Le chef de l'État dévoile des mesures pour l'économie"
        similarity = self.calculator.calculate_similarity(news1, news2)
        self.assertGreater(similarity, 0.0)

    def test_programming_languages(self):
        """Test avec des phrases sur la programmation."""
        s1 = "Python est un langage de programmation"
        s2 = "Java est un langage de programmation"
        similarity = self.calculator.calculate_similarity(s1, s2)

        # 5 mots communs (est, un, langage, de, programmation)
        # 7 mots au total
        expected = 5/7
        self.assertAlmostEqual(similarity, expected, places=3)

    def test_animal_sentences(self):
        """Test avec des phrases sur les animaux."""
        s1 = "Le chat mange des croquettes"
        s2 = "Le chien mange des os"
        similarity = self.calculator.calculate_similarity(s1, s2)

        # 3 mots communs (le, mange, des)
        # 7 mots au total
        expected = 3/7
        self.assertAlmostEqual(similarity, expected, places=3)


class TestMathematicalProperties(unittest.TestCase):
    """Vérification des propriétés mathématiques de Jaccard."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_range_property(self):
        """La similarité doit toujours être entre 0 et 1."""
        sentences = [
            "chat mange",
            "chien court",
            "oiseau vole",
            "poisson nage"
        ]

        for s1 in sentences:
            for s2 in sentences:
                similarity = self.calculator.calculate_similarity(s1, s2)
                self.assertTrue(0 <= similarity <= 1,
                                f"Similarité hors limites: {similarity}")

    def test_reflexivity(self):
        """Une phrase comparée à elle-même doit toujours donner 1."""
        sentences = ["chat", "chien court", "oiseau vole rapidement"]

        for sentence in sentences:
            similarity = self.calculator.calculate_similarity(
                sentence, sentence)
            self.assertEqual(similarity, 1.0,
                             f"Réflexivité échouée pour '{sentence}'")

    def test_symmetry(self):
        """Jaccard(A,B) doit être égal à Jaccard(B,A)."""
        pairs = [
            ("chat mange", "chien court"),
            ("python code", "java programmation"),
            ("bonjour monde", "hello world")
        ]

        for s1, s2 in pairs:
            sim1 = self.calculator.calculate_similarity(s1, s2)
            sim2 = self.calculator.calculate_similarity(s2, s1)
            self.assertAlmostEqual(sim1, sim2, places=10,
                                   msg=f"Symétrie échouée pour '{s1}' et '{s2}'")


class TestEdgeCases(unittest.TestCase):
    """Tests de cas particuliers et limites."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_only_punctuation(self):
        """Phrases composées uniquement de ponctuation."""
        similarity = self.calculator.calculate_similarity("!!!", "???")
        self.assertEqual(similarity, 0.0)

    def test_repeated_words(self):
        """Les mots répétés ne comptent qu'une fois dans les ensembles."""
        s1 = "chat chat chat"
        s2 = "chat"
        similarity = self.calculator.calculate_similarity(s1, s2)
        self.assertEqual(similarity, 1.0)

    def test_very_long_sentences(self):
        """Test avec des phrases très longues pour vérifier la robustesse."""
        long_sentence1 = " ".join(["mot"] * 100)
        long_sentence2 = " ".join(["mot"] * 50)
        similarity = self.calculator.calculate_similarity(
            long_sentence1, long_sentence2)
        self.assertEqual(similarity, 1.0)

    def test_special_characters(self):
        """Test avec des caractères spéciaux."""
        s1 = "hello@world.com"
        s2 = "hello world com"
        similarity = self.calculator.calculate_similarity(s1, s2)
        self.assertGreater(similarity, 0.0)
        self.assertAlmostEqual(similarity, 1.0, places=2)


class TestPerformance(unittest.TestCase):
    """Tests de performance du calculateur."""

    def setUp(self):
        self.calculator = JaccardSimilarity()

    def test_large_sentences_performance(self):
        """Mesure du temps de calcul avec de grandes phrases."""
        words = [f"mot{i}" for i in range(1000)]
        sentence1 = " ".join(words[:800])
        sentence2 = " ".join(words[200:])

        start_time = time.time()
        similarity = self.calculator.calculate_similarity(sentence1, sentence2)
        end_time = time.time()

        # Ça devrait prendre moins d'une seconde
        self.assertLess(end_time - start_time, 1.0)

        # On vérifie aussi que le résultat est cohérent
        self.assertTrue(0 <= similarity <= 1)

    def test_many_comparisons_performance(self):
        """Test de performance avec beaucoup de comparaisons."""
        sentences = [f"phrase numéro {i} avec des mots" for i in range(50)]

        start_time = time.time()
        results = self.calculator.compare_multiple_sentences(sentences)
        end_time = time.time()

        # On doit avoir 50*49/2 = 1225 comparaisons
        expected_comparisons = 50 * 49 // 2
        self.assertEqual(len(results), expected_comparisons)

        # Ça devrait prendre moins de 2 secondes
        self.assertLess(end_time - start_time, 2.0)


def run_performance_summary():
    """Affichage d'un résumé des performances."""
    print("\n" + "="*70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*70)

    calculator = JaccardSimilarity()

    test_sizes = [10, 50, 100, 200]

    for size in test_sizes:
        sentences = [
            f"phrase de test {i} avec quelques mots" for i in range(size)]

        start_time = time.time()
        results = calculator.compare_multiple_sentences(sentences)
        end_time = time.time()

        execution_time = end_time - start_time
        comparisons = len(results)
        comp_per_sec = comparisons / \
            execution_time if execution_time > 0 else float('inf')

        print(f"  {size:3d} phrases → {comparisons:5d} comparaisons en {execution_time:.3f}s "
              f"({comp_per_sec:.0f} comp/s)")


if __name__ == "__main__":
    print("="*70)
    print("TESTS UNITAIRES - SIMILARITÉ DE JACCARD")
    print("="*70)

    unittest.main(verbosity=2, exit=False)

    run_performance_summary()

    print("\n" + "="*70)
    print("TESTS TERMINÉS")
    print("="*70)
```

# README.md

```markdown
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
Distance de Jaccard: d(A,B) = 1 - J(A,B)

````

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
````

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

```

```

# QUICKSTART.md

````markdown
# Guide de Démarrage Rapide

## Installation en 30 secondes

```bash
# Cloner le dépôt
git clone https://github.com/[votre-username]/jaccard-similarity-project.git
cd jaccard-similarity-project

# Aucune installation requise, le projet est prêt à l'emploi
python jaccard_similarity.py
```
````

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

```

```

# .gitignore

```.gitignore
# Fichiers compilés Python
__pycache__/
*.py[cod]
*$py.class

# Extensions C
*.so

# Fichiers de distribution et packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Fichiers PyInstaller
*.manifest
*.spec

# Logs d'installation
pip-log.txt
pip-delete-this-directory.txt

# Rapports de tests et couverture
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Fichiers de traduction
*.mo
*.pot

# Fichiers Django
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Fichiers Flask
instance/
.webassets-cache

# Fichiers Scrapy
.scrapy

# Documentation Sphinx
docs/_build/

# Fichiers PyBuilder
target/

# Checkpoints Jupyter
.ipynb_checkpoints

# Configuration IPython
profile_default/
ipython_config.py

# Version Python (pyenv)
.python-version

# Lock file pipenv
Pipfile.lock

# Répertoire PEP 582
__pypackages__/

# Fichiers Celery
celerybeat-schedule
celerybeat.pid

# Fichiers parsés SageMath
*.sage.py

# Environnements virtuels
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Configuration Spyder
.spyderproject
.spyproject

# Configuration Rope
.ropeproject

# Documentation mkdocs
/site

# Cache mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Vérificateur de types Pyre
.pyre/

# Fichiers VSCode
.vscode/
*.code-workspace

# Fichiers PyCharm
.idea/
*.iml
*.iws

# Fichiers Sublime Text
*.sublime-project
*.sublime-workspace

# Fichiers système Mac
.DS_Store

# Fichiers système Windows
Thumbs.db
ehthumbs.db
Desktop.ini

# Fichiers temporaires
*.tmp
*.temp
*~

# Dossiers et fichiers spécifiques au projet
results/
output/
data/
*.csv
*.xlsx
*.json
*.db
logs/
```

```

```
