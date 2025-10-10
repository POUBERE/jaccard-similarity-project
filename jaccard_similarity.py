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
