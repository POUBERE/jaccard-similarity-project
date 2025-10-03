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

    def interpret_similarity(self, similarity: float, context: str = "general") -> Dict[str, str]:
        """
        Interprète un score de similarité de Jaccard de manière détaillée.

        Args:
            similarity (float): Score de similarité entre 0 et 1
            context (str): Contexte d'utilisation ('general', 'plagiarism', 'clustering', 'search')

        Returns:
            Dict[str, str]: Dictionnaire contenant l'interprétation détaillée
        """
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
            category = "Moyennement similaire"
            emoji = "🟠"
            color_code = "orange"
        elif similarity >= 0.2:
            category = "Peu similaire"
            emoji = "🔴"
            color_code = "red"
        elif similarity > 0:
            category = "Très peu similaire"
            emoji = "⚫"
            color_code = "dark_red"
        else:
            category = "Aucune similarité"
            emoji = "❌"
            color_code = "black"

        # Interprétation générale
        general_interpretation = self._get_general_interpretation(similarity)

        # Interprétation contextuelle
        contextual_interpretation = self._get_contextual_interpretation(
            similarity, context)

        # Recommandations
        recommendations = self._get_recommendations(similarity, context)

        # Explication technique
        technical_explanation = self._get_technical_explanation(similarity)

        return {
            'score': similarity,
            'category': category,
            'emoji': emoji,
            'color_code': color_code,
            'general_interpretation': general_interpretation,
            'contextual_interpretation': contextual_interpretation,
            'recommendations': recommendations,
            'technical_explanation': technical_explanation
        }

    def _get_general_interpretation(self, similarity: float) -> str:
        """Fournit une interprétation générale du score."""
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

    def _get_contextual_interpretation(self, similarity: float, context: str) -> str:
        """Fournit une interprétation selon le contexte d'utilisation."""
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

        # Sélection de l'interprétation appropriée
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

    def _get_recommendations(self, similarity: float, context: str) -> List[str]:
        """Fournit des recommandations basées sur le score et le contexte."""
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

        elif context == 'search':
            if similarity >= 0.4:
                recommendations.append(
                    "Document pertinent, à inclure dans les résultats")
            else:
                recommendations.append(
                    "Document peu pertinent, peut être exclu des résultats")

        # Recommandations générales selon le score
        if similarity == 0.0:
            recommendations.append(
                "Aucun mot commun - Vérifier le prétraitement des textes")
        elif similarity < 0.3 and len(recommendations) == 0:
            recommendations.append(
                "Similarité faible - Ces textes traitent probablement de sujets différents")

        return recommendations if recommendations else ["Aucune recommandation spécifique"]

    def _get_technical_explanation(self, similarity: float) -> str:
        """Fournit une explication technique du score."""
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

        return explanation


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

    print("1. Tests de base avec interprétations :")
    print("-" * 80)
    for i, (s1, s2) in enumerate(examples, 1):
        similarity = calculator.calculate_similarity(s1, s2)
        interpretation = calculator.interpret_similarity(
            similarity, context='general')

        print(f"\nTest {i}:")
        print(f"  Phrase 1: '{s1}'")
        print(f"  Phrase 2: '{s2}'")
        print(f"  Similarité: {similarity:.4f}")
        print(
            f"  Catégorie: {interpretation['emoji']} {interpretation['category']}")
        print(f"\n  💡 Interprétation:")
        print(f"     {interpretation['general_interpretation']}")
        print(f"\n  📊 Contexte:")
        print(f"     {interpretation['contextual_interpretation']}")
        print("-" * 80)

    print("\n2. Analyse détaillée avec recommandations :")
    print("-" * 80)
    detailed = calculator.calculate_similarity_detailed(
        "Le chat mange", "Le chien mange")

    similarity = detailed['jaccard_similarity']
    interpretation = calculator.interpret_similarity(
        similarity, context='general')

    print(f"Phrase 1: '{detailed['sentence1']}'")
    print(f"Phrase 2: '{detailed['sentence2']}'")
    print(f"\nMots phrase 1: {sorted(detailed['words_set1'])}")
    print(f"Mots phrase 2: {sorted(detailed['words_set2'])}")
    print(f"\nIntersection (mots communs): {sorted(detailed['intersection'])}")
    print(f"Union (tous les mots): {sorted(detailed['union'])}")
    print(f"\nTaille intersection: {detailed['intersection_size']}")
    print(f"Taille union: {detailed['union_size']}")
    print(f"\nSimilarité Jaccard: {similarity:.4f}")
    print(f"Catégorie: {interpretation['emoji']} {interpretation['category']}")

    print(f"\n📖 Explication technique:")
    print(f"   {interpretation['technical_explanation']}")

    print(f"\n💡 Interprétation générale:")
    print(f"   {interpretation['general_interpretation']}")

    print(f"\n📌 Recommandations:")
    for rec in interpretation['recommendations']:
        print(f"   • {rec}")
    print("-" * 80)

    print("\n3. Matrice de similarité avec interprétations :")
    print("-" * 80)
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

    # Interprétations des relations
    print("\n📊 Interprétations des relations:")
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            sim = matrix[i][j]
            interp = calculator.interpret_similarity(sim, context='clustering')
            print(
                f"\nPhrases {i} ↔ {j}: {sim:.4f} - {interp['emoji']} {interp['category']}")
            print(f"  {interp['contextual_interpretation']}")


def interactive_mode(calculator: JaccardSimilarity):
    """
    Mode interactif pour saisir des phrases manuellement.

    Args:
        calculator (JaccardSimilarity): Instance du calculateur
    """
    print("=== Mode Interactif - Calculateur de Similarité de Jaccard ===")
    print("Entrez 'quit' pour quitter\n")

    while True:
        sentence1 = input("Phrase 1: ").strip()
        if sentence1.lower() == 'quit':
            break

        sentence2 = input("Phrase 2: ").strip()
        if sentence2.lower() == 'quit':
            break

        similarity = calculator.calculate_similarity(sentence1, sentence2)
        interpretation = calculator.interpret_similarity(
            similarity, context='general')

        print(f"\n{'='*70}")
        print(f"RÉSULTAT DE LA COMPARAISON")
        print(f"{'='*70}")
        print(f"\nSimilarité de Jaccard: {similarity:.4f}")
        print(
            f"Catégorie: {interpretation['emoji']} {interpretation['category']}")

        # Affichage des informations complémentaires
        set1 = calculator.preprocess_sentence(sentence1)
        set2 = calculator.preprocess_sentence(sentence2)
        intersection = set1.intersection(set2)

        print(
            f"\nMots communs: {sorted(intersection)} ({len(intersection)} mots)")
        print(f"Total mots uniques: {len(set1.union(set2))} mots")

        print(f"\n💡 Interprétation:")
        print(f"   {interpretation['general_interpretation']}")

        print(f"\n📖 Explication technique:")
        for line in interpretation['technical_explanation'].split('\n'):
            print(f"   {line}")

        print(f"\n📌 Recommandations:")
        for rec in interpretation['recommendations']:
            print(f"   • {rec}")

        print("-" * 70)
        print()


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Calcul de similarité de Jaccard entre phrases',
        epilog='Exemple: python jaccard_similarity.py --interactive'
    )
    parser.add_argument('--case-sensitive', action='store_true',
                        help='Respecte la casse des mots')
    parser.add_argument('--keep-punctuation', action='store_true',
                        help='Garde la ponctuation')
    parser.add_argument('--interactive', action='store_true',
                        help='Mode interactif pour saisir des phrases')

    args = parser.parse_args()

    # Configuration du calculateur selon les arguments
    calculator = JaccardSimilarity(
        case_sensitive=args.case_sensitive,
        remove_punctuation=not args.keep_punctuation
    )

    if args.interactive:
        interactive_mode(calculator)
    else:
        run_example_tests(calculator)


if __name__ == "__main__":
    main()
