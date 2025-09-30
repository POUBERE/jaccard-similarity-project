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