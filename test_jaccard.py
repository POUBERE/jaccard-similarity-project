#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires
Projet de Machine Learning non Supervisé

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Novembre 2025
"""

import unittest
from french_synonyms import FrenchSynonyms
from french_lemmatizer import FrenchLemmatizer
from semantic_analyzer import SemanticAnalyzer
from jaccard_similarity_v3 import JaccardSimilarity


class TestFrenchSynonyms(unittest.TestCase):
    """Tests pour le module de synonymes."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.synonyms = FrenchSynonyms()

    def test_get_synonyms(self):
        """Test de récupération des synonymes."""
        syns = self.synonyms.get_synonyms('chat')
        self.assertIn('chat', syns)
        self.assertIn('félin', syns)
        self.assertIn('minet', syns)

    def test_are_synonyms(self):
        """Test de vérification de synonymes."""
        self.assertTrue(self.synonyms.are_synonyms('voiture', 'automobile'))
        self.assertTrue(self.synonyms.are_synonyms('chat', 'félin'))
        self.assertFalse(self.synonyms.are_synonyms('chat', 'chien'))

    def test_expand_with_synonyms(self):
        """Test d'expansion avec synonymes."""
        words = {'chat'}
        expanded = self.synonyms.expand_with_synonyms(words)
        self.assertGreater(len(expanded), len(words))
        self.assertIn('chat', expanded)
        self.assertIn('félin', expanded)

    def test_get_common_synonyms(self):
        """Test de mots communs avec synonymes."""
        set1 = {'chat'}
        set2 = {'félin'}
        common = self.synonyms.get_common_synonyms(set1, set2)
        # chat et félin sont synonymes, donc il devrait y avoir une intersection
        self.assertGreater(len(common), 0)

    def test_add_custom_synonyms(self):
        """Test d'ajout de synonymes personnalisés."""
        custom = {'ia', 'intelligence artificielle', 'ai'}
        self.synonyms.add_custom_synonyms(custom)
        self.assertTrue(self.synonyms.are_synonyms('ia', 'ai'))


class TestFrenchLemmatizer(unittest.TestCase):
    """Tests pour le lemmatiseur."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.lemmatizer = FrenchLemmatizer()

    def test_lemmatize_verb_etre(self):
        """Test de lemmatisation du verbe être."""
        self.assertEqual(self.lemmatizer.lemmatize('suis'), 'être')
        self.assertEqual(self.lemmatizer.lemmatize('étais'), 'être')
        self.assertEqual(self.lemmatizer.lemmatize('serai'), 'être')

    def test_lemmatize_verb_avoir(self):
        """Test de lemmatisation du verbe avoir."""
        self.assertEqual(self.lemmatizer.lemmatize('ai'), 'avoir')
        self.assertEqual(self.lemmatizer.lemmatize('avais'), 'avoir')
        self.assertEqual(self.lemmatizer.lemmatize('aurai'), 'avoir')

    def test_lemmatize_verb_aller(self):
        """Test de lemmatisation du verbe aller."""
        self.assertEqual(self.lemmatizer.lemmatize('vais'), 'aller')
        self.assertEqual(self.lemmatizer.lemmatize('allais'), 'aller')
        self.assertEqual(self.lemmatizer.lemmatize('irai'), 'aller')

    def test_lemmatize_regular_verb(self):
        """Test de lemmatisation de verbes réguliers."""
        lemma = self.lemmatizer.lemmatize('mange')
        self.assertEqual(lemma, 'manger')

        lemma = self.lemmatizer.lemmatize('mangeons')
        self.assertEqual(lemma, 'manger')

    def test_lemmatize_plural_noun(self):
        """Test de lemmatisation de noms au pluriel."""
        self.assertEqual(self.lemmatizer.lemmatize('chevaux'), 'cheval')
        self.assertEqual(self.lemmatizer.lemmatize('animaux'), 'animal')
        self.assertEqual(self.lemmatizer.lemmatize('bateaux'), 'bateau')
        self.assertEqual(self.lemmatizer.lemmatize('chats'), 'chat')

    def test_lemmatize_feminine_adjective(self):
        """Test de lemmatisation d'adjectifs féminins."""
        self.assertEqual(self.lemmatizer.lemmatize('belle'), 'beau')
        self.assertEqual(self.lemmatizer.lemmatize('bonne'), 'bon')
        self.assertEqual(self.lemmatizer.lemmatize('grande'), 'grand')

    def test_add_custom_lemma(self):
        """Test d'ajout de lemme personnalisé."""
        self.lemmatizer.add_custom_lemma('tweets', 'tweet')
        self.assertEqual(self.lemmatizer.lemmatize('tweets'), 'tweet')


class TestSemanticAnalyzer(unittest.TestCase):
    """Tests pour l'analyseur sémantique."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.analyzer = SemanticAnalyzer()

    def test_get_semantic_fields(self):
        """Test de récupération des champs sémantiques."""
        fields = self.analyzer.get_semantic_fields('chat')
        self.assertIn('animaux', fields)

        fields = self.analyzer.get_semantic_fields('voiture')
        self.assertIn('véhicules', fields)

    def test_are_semantically_related(self):
        """Test de relation sémantique."""
        # Chat et chien sont tous deux dans le champ "animaux"
        self.assertTrue(self.analyzer.are_semantically_related('chat', 'chien'))

        # Chat et voiture ne sont pas dans le même champ
        self.assertFalse(self.analyzer.are_semantically_related('chat', 'voiture'))

    def test_semantic_similarity(self):
        """Test de similarité sémantique."""
        # Même mot
        sim = self.analyzer.semantic_similarity('chat', 'chat')
        self.assertEqual(sim, 1.0)

        # Mots du même champ
        sim = self.analyzer.semantic_similarity('chat', 'chien')
        self.assertGreater(sim, 0.0)

        # Antonymes
        sim = self.analyzer.semantic_similarity('grand', 'petit')
        self.assertEqual(sim, 0.0)

    def test_get_related_words(self):
        """Test de recherche de mots liés."""
        related = self.analyzer.get_related_words('chat', max_words=5)
        self.assertGreater(len(related), 0)
        self.assertLessEqual(len(related), 5)

        # Vérifier que ce sont des tuples (mot, score)
        for word, score in related:
            self.assertIsInstance(word, str)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    def test_semantic_sentence_similarity(self):
        """Test de similarité sémantique de phrases."""
        set1 = {'chat', 'mange'}
        set2 = {'chien', 'dévore'}

        sim = self.analyzer.semantic_sentence_similarity(set1, set2)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_add_semantic_field(self):
        """Test d'ajout de champ sémantique personnalisé."""
        custom_field = {'python', 'java', 'javascript', 'ruby'}
        self.analyzer.add_semantic_field('langages', custom_field)

        fields = self.analyzer.get_semantic_fields('python')
        self.assertIn('langages', fields)


class TestJaccardSimilarity(unittest.TestCase):
    """Tests pour la classe JaccardSimilarity."""

    def test_basic_similarity_no_options(self):
        """Test de similarité basique sans options."""
        calc = JaccardSimilarity()

        sim = calc.calculate_similarity("Le chat mange", "Le chat mange")
        self.assertEqual(sim, 1.0)

        sim = calc.calculate_similarity("Le chat mange", "Le chien court")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_with_lemmatization(self):
        """Test avec lemmatisation."""
        calc = JaccardSimilarity(use_lemmatization=True)

        # "suis" et "être" devraient être traités comme similaires
        sim = calc.calculate_similarity("Je suis content", "Être content")
        self.assertGreater(sim, 0.0)

    def test_with_synonyms(self):
        """Test avec gestion des synonymes."""
        calc = JaccardSimilarity(
            remove_stopwords=True,
            use_synonyms=True
        )

        # "chat" et "félin" sont synonymes
        sim = calc.calculate_similarity("chat noir", "félin blanc")
        self.assertGreater(sim, 0.0)

    def test_with_semantic_analysis(self):
        """Test avec analyse sémantique."""
        calc = JaccardSimilarity(
            remove_stopwords=True,
            use_semantic_analysis=True
        )

        result = calc.calculate_similarity_detailed("chat mange", "chien court")

        # Devrait avoir une clé semantic_similarity
        self.assertIn('semantic_similarity', result)
        self.assertIn('hybrid_similarity', result)

    def test_full_configuration(self):
        """Test avec toutes les options activées."""
        calc = JaccardSimilarity(
            remove_stopwords=True,
            use_lemmatization=True,
            use_synonyms=True,
            use_semantic_analysis=True
        )

        s1 = "Le chat mange une souris"
        s2 = "Le félin dévore un rat"

        result = calc.calculate_similarity_detailed(s1, s2)

        # Vérifier toutes les clés attendues
        self.assertIn('jaccard_similarity', result)
        self.assertIn('semantic_similarity', result)
        self.assertIn('hybrid_similarity', result)
        self.assertIn('common_via_synonyms', result)

        # La similarité devrait être significative
        self.assertGreater(result['jaccard_similarity'], 0.5)

    def test_hybrid_similarity(self):
        """Test de la similarité hybride."""
        calc = JaccardSimilarity(
            remove_stopwords=True,
            use_semantic_analysis=True
        )

        hybrid = calc.calculate_hybrid_similarity("chat noir", "chien blanc")

        # Devrait être un float entre 0 et 1
        self.assertIsInstance(hybrid, float)
        self.assertGreaterEqual(hybrid, 0.0)
        self.assertLessEqual(hybrid, 1.0)

    def test_comparison_v2_vs_v3(self):
        """Test de comparaison v2.0 vs v3.0."""
        # Configuration v2.0
        calc_v2 = JaccardSimilarity(
            remove_stopwords=True,
            use_stemming=True
        )

        # Configuration v3.0
        calc_v3 = JaccardSimilarity(
            remove_stopwords=True,
            use_lemmatization=True,
            use_synonyms=True
        )

        s1 = "Le chat mange une souris"
        s2 = "Le félin dévore un rat"

        sim_v2 = calc_v2.calculate_similarity(s1, s2)
        sim_v3 = calc_v3.calculate_similarity(s1, s2)

        # v3.0 devrait être significativement meilleure
        self.assertGreater(sim_v3, sim_v2)
        self.assertGreater(sim_v3, 0.5)

    def test_export_json(self):
        """Test d'export JSON."""
        calc = JaccardSimilarity(
            use_lemmatization=True,
            use_synonyms=True
        )

        results = [
            calc.calculate_similarity_detailed("chat noir", "félin blanc")
        ]

        filename = calc.export_results_to_json(results, "test_export_v3.json")

        self.assertIsNotNone(filename)

        # Nettoyer
        import os
        if filename and os.path.exists(filename):
            os.remove(filename)

    def test_get_config_summary(self):
        """Test du résumé de configuration."""
        calc = JaccardSimilarity(
            use_lemmatization=True,
            use_synonyms=True,
            use_semantic_analysis=True
        )

        summary = calc.get_config_summary()

        self.assertIsInstance(summary, str)
        self.assertIn('v3.0', summary.lower())


def run_tests():
    """Lance tous les tests."""
    print("=" * 80)
    print("TESTS UNITAIRES - VERSION 3.0")
    print("=" * 80)
    print()

    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestFrenchSynonyms))
    suite.addTests(loader.loadTestsFromTestCase(TestFrenchLemmatizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestJaccardSimilarity))

    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Afficher le résumé
    print()
    print("=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Réussites: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print()

    if result.wasSuccessful():
        print("[OK] TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    else:
        print("[ERREUR] Certains tests ont échoué.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
