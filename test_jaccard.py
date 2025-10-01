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