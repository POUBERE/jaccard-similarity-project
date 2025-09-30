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