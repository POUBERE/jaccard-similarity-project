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

