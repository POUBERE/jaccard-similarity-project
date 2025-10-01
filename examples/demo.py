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
