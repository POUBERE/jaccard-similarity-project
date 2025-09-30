#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de calcul de similarité de Jaccard entre phrases
Projet de Machine Learning non Supervisé

Auteurs: [Votre groupe]
Date: Septembre 2025

La similarité de Jaccard mesure la ressemblance entre deux ensembles
en calculant le rapport entre l'intersection et l'union des ensembles.
Formule: Jaccard(A,B) = |A ∩ B| / |A ∪ B|
"""

import re
import argparse
from typing import Set, List, Tuple, Dict


class JaccardSimilarity:
    """
    Classe pour calculer la similarité de Jaccard entre phrases.

    Attributes:
        case_sensitive (bool): Si True, respecte la casse des mots
        remove_punctuation (bool): Si True, supprime la ponctuation
    """

    def __init__(self, case_sensitive: bool = False, remove_punctuation: bool = True):
        """
        Initialise le calculateur de similarité.

        Args:
            case_sensitive (bool): Si True, respecte la casse des mots
            remove_punctuation (bool): Si True, supprime la ponctuation
        """
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation

    def preprocess_sentence(self, sentence: str) -> Set[str]:
        """
        Prétraite une phrase en la convertissant en ensemble de mots.

        Args:
            sentence (str): La phrase à prétraiter

        Returns:
            Set[str]: Ensemble des mots de la phrase

        Exemple:
            >>> calc = JaccardSimilarity()
            >>> calc.preprocess_sentence("Hello, World!")
            {'hello', 'world'}
        """
        # Conversion en minuscules si nécessaire
        if not self.case_sensitive:
            sentence = sentence.lower()

        # Nettoyage de la ponctuation
        if self.remove_punctuation:
            sentence = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', sentence)

        # Séparation en mots individuels et création de l'ensemble
        words = set(word.strip() for word in sentence.split() if word.strip())

        return words

    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Calcule la similarité de Jaccard entre deux phrases.

        Args:
            sentence1 (str): Première phrase
            sentence2 (str): Deuxième phrase

        Returns:
            float: Similarité de Jaccard (entre 0 et 1)

        Exemple:
            >>> calc = JaccardSimilarity()
            >>> calc.calculate_similarity("Le chat mange", "Le chien mange")
            0.6666666666666666
        """
        # Prétraitement des deux phrases
        set1 = self.preprocess_sentence(sentence1)
        set2 = self.preprocess_sentence(sentence2)

        # Calcul de l'intersection et de l'union
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Gestion du cas où l'union est vide
        if len(union) == 0:
            return 0.0

        # Application de la formule de Jaccard
        similarity = len(intersection) / len(union)

        return similarity

    def calculate_similarity_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la similarité de Jaccard avec les détails du calcul.

        Args:
            sentence1 (str): Première phrase
            sentence2 (str): Deuxième phrase

        Returns:
            dict: Dictionnaire contenant les détails du calcul incluant les ensembles,
                  l'intersection, l'union et la similarité
        """
        set1 = self.preprocess_sentence(sentence1)
        set2 = self.preprocess_sentence(sentence2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

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

    def compare_multiple_sentences(self, sentences: List[str]) -> List[Tuple[int, int, float]]:
        """
        Compare toutes les paires de phrases dans une liste.

        Args:
            sentences (List[str]): Liste des phrases à comparer

        Returns:
            List[Tuple[int, int, float]]: Liste de tuples (index1, index2, similarité)
        """
        results = []

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self.calculate_similarity(
                    sentences[i], sentences[j])
                results.append((i, j, similarity))

        return results

    def get_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """
        Calcule la matrice de similarité pour une liste de phrases.

        Args:
            sentences (List[str]): Liste des phrases à comparer

        Returns:
            List[List[float]]: Matrice de similarité carrée
        """
        n = len(sentences)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.calculate_similarity(
                        sentences[i], sentences[j])

        return matrix

    def get_most_similar_pair(self, sentences: List[str]) -> Tuple[int, int, float]:
        """
        Trouve la paire de phrases la plus similaire.

        Args:
            sentences (List[str]): Liste des phrases à comparer

        Returns:
            Tuple[int, int, float]: (index1, index2, similarité_max)
        """
        comparisons = self.compare_multiple_sentences(sentences)

        if not comparisons:
            return (0, 0, 0.0)

        return max(comparisons, key=lambda x: x[2])


def run_example_tests(calculator: JaccardSimilarity):
    """
    Exécute des tests d'exemple pour démontrer le fonctionnement.

    Args:
        calculator (JaccardSimilarity): Instance du calculateur
    """
    print("=== Programme de Calcul de Similarité de Jaccard ===\n")

    # Définition des cas de test
    examples = [
        ("Le chat mange des croquettes", "Le chien mange des os"),
        ("Python est un langage de programmation",
         "Java est un langage de programmation"),
        ("Machine learning supervisé", "Apprentissage automatique supervisé"),
        ("Bonjour tout le monde", "Salut tout le monde"),
        ("Aucun mot en commun", "Différentes phrases complètement")
    ]

    print("1. Tests de base :")
    print("-" * 60)
    for i, (s1, s2) in enumerate(examples, 1):
        similarity = calculator.calculate_similarity(s1, s2)
        print(f"Test {i}:")
        print(f"  Phrase 1: '{s1}'")
        print(f"  Phrase 2: '{s2}'")
        print(f"  Similarité: {similarity:.4f}")
        print()

    print("2. Analyse détaillée d'un exemple :")
    print("-" * 60)
    detailed = calculator.calculate_similarity_detailed(
        "Le chat mange", "Le chien mange")
    print(f"Phrase 1: '{detailed['sentence1']}'")
    print(f"Phrase 2: '{detailed['sentence2']}'")
    print(f"Mots phrase 1: {sorted(detailed['words_set1'])}")
    print(f"Mots phrase 2: {sorted(detailed['words_set2'])}")
    print(f"Intersection: {sorted(detailed['intersection'])}")
    print(f"Union: {sorted(detailed['union'])}")
    print(f"Taille intersection: {detailed['intersection_size']}")
    print(f"Taille union: {detailed['union_size']}")
    print(f"Similarité Jaccard: {detailed['jaccard_similarity']:.4f}")
    print()

    print("3. Matrice de similarité :")
    print("-" * 60)
    test_sentences = [
        "Le chat mange",
        "Le chien mange",
        "Les animaux mangent",
        "Python est génial"
    ]

    matrix = calculator.get_similarity_matrix(test_sentences)

    print("Phrases testées :")
    for i, sentence in enumerate(test_sentences):
        print(f"  {i}: '{sentence}'")
    print()

    print("Matrice de similarité :")
    print("     ", end="")
    for i in range(len(test_sentences)):
        print(f"{i:8}", end="")
    print()

    for i, row in enumerate(matrix):
        print(f"{i}: ", end="")
        for similarity in row:
            print(f"{similarity:8.4f}", end="")
        print()


