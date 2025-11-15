#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de calcul de similarité de Jaccard entre phrases
Projet de Machine Learning non Supervisé

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Novembre 2025
- Lemmatisation avancée (amélioration du stemming)
- Analyse sémantique basique
- MODE INTERACTIF avec configuration complète
"""

import re
import argparse
import json
from typing import Set, List, Tuple, Dict
from datetime import datetime

# Import des nouveaux modules v3.0
from french_synonyms import FrenchSynonyms
from french_lemmatizer import FrenchLemmatizer
from semantic_analyzer import SemanticAnalyzer

# ============================================================================
# CLASSE FrenchStemmer (pour compatibilité avec v2.0)
# ============================================================================

class FrenchStemmer:
    """Stemmer pour le français avec gestion des cas spéciaux."""

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

    PROTECTED_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'de', 'du', 'au', 'aux', 'ce', 'ces',
        'et', 'ou', 'mais', 'car', 'or', 'donc', 'ni',
        'si', 'ne', 'pas', 'plus', 'très', 'bien', 'tout'
    }

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
        """Applique le stemming à un mot français."""
        if len(word) <= 2:
            return word.lower()

        word_lower = word.lower()

        if word_lower in FrenchStemmer.PROTECTED_WORDS:
            return word_lower

        if word_lower in FrenchStemmer.EXCEPTIONS:
            return FrenchStemmer.EXCEPTIONS[word_lower]

        for suffix in FrenchStemmer.SUFFIXES:
            if word_lower.endswith(suffix):
                stem_candidate = word_lower[:-len(suffix)]
                if len(stem_candidate) >= 3:
                    return stem_candidate
                break

        return word_lower


# ============================================================================
# CLASSE JaccardSimilarity
# ============================================================================

class JaccardSimilarity:
    """
    Classe améliorée pour calculer la similarité de Jaccard entre phrases.

    - use_synonyms: Gestion des synonymes
    - use_lemmatization: Lemmatisation avancée au lieu du stemming basique
    - use_semantic_analysis: Analyse sémantique pour liens conceptuels
    """

    # Stop-words français
    FRENCH_STOPWORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
        'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
        'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'lui', 'y', 'en',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
        'à', 'dans', 'par', 'pour', 'en', 'vers', 'avec', 'sans', 'sous', 'sur',
        'qui', 'que', 'quoi', 'dont', 'où',
        'si', 'ne', 'pas', 'plus', 'moins', 'très', 'tout', 'toute', 'tous', 'toutes'
    }

    def __init__(self,
                 case_sensitive: bool = False,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = False,
                 use_stemming: bool = False,
                 use_lemmatization: bool = False,
                 use_synonyms: bool = False,
                 use_semantic_analysis: bool = False):
        """
        Initialise le calculateur avec les options choisies.

        Paramètres:
            case_sensitive: Si True, "Python" et "python" sont différents
            remove_punctuation: Si True, enlève la ponctuation
            remove_stopwords: Si True, filtre les stop-words
            use_stemming: Si True, applique le stemming
            use_lemmatization: Si True, applique la lemmatisation
            use_synonyms: Si True, gère les synonymes
            use_semantic_analysis: Si True, analyse sémantique

        Note: Si use_lemmatization est True, use_stemming est ignoré
        """
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.use_synonyms = use_synonyms
        self.use_semantic_analysis = use_semantic_analysis

        # Initialiser les outils selon les options
        self.stemmer = FrenchStemmer() if use_stemming and not use_lemmatization else None
        self.lemmatizer = FrenchLemmatizer() if use_lemmatization else None
        self.synonyms = FrenchSynonyms() if use_synonyms else None
        self.semantic = SemanticAnalyzer() if use_semantic_analysis else None

    def preprocess_sentence(self, sentence: str) -> Set[str]:
        """
        Prétraite une phrase et la convertit en ensemble de mots.

        Étapes du traitement:
            1. Normalisation de la casse
            2. Suppression de la ponctuation
            3. Découpage en mots
            4. Filtrage des stop-words
            5. Lemmatisation OU Stemming
            6. Expansion avec synonymes (si activé)
        """
        # Normalisation de la casse
        if not self.case_sensitive:
            sentence = sentence.lower()

        # Suppression de la ponctuation
        if self.remove_punctuation:
            sentence = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', sentence)

        # Découpage en mots
        words = [word.strip() for word in sentence.split() if word.strip()]

        # Filtrage des stop-words
        if self.remove_stopwords:
            words = [w for w in words if w.lower() not in self.FRENCH_STOPWORDS]

        # Lemmatisation OU Stemming (priorité à la lemmatisation)
        if self.use_lemmatization and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        elif self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(w) for w in words]

        # Conversion en Set
        word_set = set(words)

        # Expansion avec synonymes
        if self.use_synonyms and self.synonyms:
            word_set = self.synonyms.expand_with_synonyms(word_set)

        return word_set

    def calculate_similarity_detailed(self, sentence1: str, sentence2: str) -> Dict:
        """
        Calcule la similarité de Jaccard avec tous les détails.

        Inclut des informations sur les synonymes
        et la similarité sémantique.
        """
        # Prétraitement
        set1 = self.preprocess_sentence(sentence1)
        set2 = self.preprocess_sentence(sentence2)

        # Calcul de l'intersection et l'union
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Calcul de la similarité de Jaccard classique
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

        result = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'words_set1': set1,
            'words_set2': set2,
            'intersection': intersection,
            'union': union,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'jaccard_similarity': jaccard_similarity,
            'jaccard_distance': 1.0 - jaccard_similarity
        }

        # Ajouter la similarité sémantique si activée
        if self.use_semantic_analysis and self.semantic:
            semantic_sim = self.semantic.semantic_sentence_similarity(set1, set2)
            result['semantic_similarity'] = semantic_sim

            # Similarité hybride (moyenne pondérée)
            hybrid_sim = (0.6 * jaccard_similarity + 0.4 * semantic_sim)
            result['hybrid_similarity'] = hybrid_sim

        # Informations sur les synonymes
        if self.use_synonyms and self.synonyms:
            # Compter les mots communs via synonymes
            common_via_synonyms = self.synonyms.get_common_synonyms(set1, set2)
            result['common_via_synonyms'] = common_via_synonyms
            result['common_via_synonyms_count'] = len(common_via_synonyms)

        return result

    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """Version simple qui retourne juste le score de similarité."""
        result = self.calculate_similarity_detailed(sentence1, sentence2)
        return result['jaccard_similarity']

    def get_config_summary(self) -> str:
        """Retourne un résumé de la configuration active."""
        features = []

        if self.use_lemmatization:
            features.append("Lemmatisation avancée")
        elif self.use_stemming:
            features.append("Stemming basique")

        if self.use_synonyms:
            features.append("Gestion des synonymes")

        if self.use_semantic_analysis:
            features.append("Analyse sémantique")

        if self.remove_stopwords:
            features.append("Filtrage stop-words")

        if not features:
            return "Configuration basique"

        return f"Configuration : {', '.join(features)}"


# ============================================================================
# MODE INTERACTIF 
# ============================================================================

def interactive_mode(calculator: JaccardSimilarity):
    """
    Mode interactif pour saisir des phrases manuellement.
    
    L'utilisateur peut saisir ses propres phrases et voir les résultats.
    Taper 'quit' pour sortir.
    """
    print("=" * 80)
    print("MODE INTERACTIF - CALCULATEUR DE JACCARD")
    print("=" * 80)
    print()
    print("Configuration active:")
    print(f"  - Sensibilité à la casse: {'Activée' if calculator.case_sensitive else 'Désactivée'}")
    print(f"  - Suppression ponctuation: {'Activée' if calculator.remove_punctuation else 'Désactivée'}")
    print(f"  - Stop-words: {'Activés' if calculator.remove_stopwords else 'Désactivés'}")
    print(f"  - Stemming : {'Activé' if calculator.use_stemming else 'Désactivé'}")
    print(f"  - Lemmatisation : {'Activée' if calculator.use_lemmatization else 'Désactivée'}")
    print(f"  - Synonymes : {'Activés' if calculator.use_synonyms else 'Désactivés'}")
    print(f"  - Analyse sémantique : {'Activée' if calculator.use_semantic_analysis else 'Désactivée'}")
    print()
    print("💡 Entrez 'quit' pour quitter")
    print("=" * 80)
    print()

    while True:
        try:
            # Saisie de la première phrase
            sentence1 = input("Phrase 1: ").strip()
            if sentence1.lower() == 'quit':
                print("\n👋 Au revoir !")
                break

            # Saisie de la deuxième phrase
            sentence2 = input("Phrase 2: ").strip()
            if sentence2.lower() == 'quit':
                print("\n👋 Au revoir !")
                break

            if not sentence1 or not sentence2:
                print("\n❌ Erreur: Les deux phrases doivent être non vides\n")
                continue

            # Calcul détaillé
            result = calculator.calculate_similarity_detailed(sentence1, sentence2)
            
            print("\n" + "=" * 80)
            print("RÉSULTAT DE LA COMPARAISON")
            print("=" * 80)

            # Affichage des ensembles de mots
            print(f"\n📝 Mots après prétraitement:")
            print(f"   Phrase 1 ({len(result['words_set1'])} mots): {sorted(result['words_set1'])}")
            print(f"   Phrase 2 ({len(result['words_set2'])} mots): {sorted(result['words_set2'])}")
            
            print(f"\n🔤 Analyse des ensembles:")
            print(f"   ∩ Intersection ({result['intersection_size']} mots): {sorted(result['intersection'])}")
            print(f"   ∪ Union ({result['union_size']} mots): {sorted(result['union'])}")

            # Affichage de la similarité Jaccard
            similarity = result['jaccard_similarity']
            print(f"\n{'─' * 80}")
            print("📊 SIMILARITÉ DE JACCARD")
            print("─" * 80)
            print(f"Score: {similarity:.4f} ({similarity*100:.2f}%)")
            
            if similarity == 1.0:
                category = "✅ Identiques"
            elif similarity >= 0.8:
                category = "🟢 Très similaires"
            elif similarity >= 0.6:
                category = "🟡 Assez similaires"
            elif similarity >= 0.4:
                category = "🟠 Moyennement similaires"
            elif similarity >= 0.2:
                category = "🔴 Peu similaires"
            elif similarity > 0:
                category = "⚫ Très peu similaires"
            else:
                category = "❌ Aucune similarité"
            
            print(f"Catégorie: {category}")

            # Affichage de la similarité sémantique
            if calculator.use_semantic_analysis and 'semantic_similarity' in result:
                print(f"\n{'─' * 80}")
                print("🧠 SIMILARITÉ SÉMANTIQUE")
                print("─" * 80)
                print(f"Score: {result['semantic_similarity']:.4f} ({result['semantic_similarity']*100:.2f}%)")
                
                if 'hybrid_similarity' in result:
                    print(f"\n{'─' * 80}")
                    print("⚖️  SIMILARITÉ HYBRIDE (Jaccard + Sémantique)")
                    print("─" * 80)
                    print(f"Score: {result['hybrid_similarity']:.4f} ({result['hybrid_similarity']*100:.2f}%)")

            # Affichage des synonymes détectés
            if calculator.use_synonyms and 'common_via_synonyms' in result:
                print(f"\n{'─' * 80}")
                print("🔄 MOTS COMMUNS VIA SYNONYMES")
                print("─" * 80)
                print(f"Nombre: {result['common_via_synonyms_count']}")
                if result['common_via_synonyms_count'] > 0:
                    print(f"Mots: {sorted(result['common_via_synonyms'])}")

            # Formule mathématique
            print(f"\n{'─' * 80}")
            print("📐 FORMULE")
            print("─" * 80)
            print(f"Similarité(A,B) = |A ∩ B| / |A ∪ B|")
            print(f"                = {result['intersection_size']} / {result['union_size']}")
            print(f"                = {similarity:.4f}")
            
            print("\n" + "=" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Programme interrompu. Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")


# ============================================================================
# DÉMONSTRATION COMPARAISON V2.0 vs V3.0
# ============================================================================

def run_comparison_v2_v3():
    """
    Démontre la différence entre la v2.0 et la v3.0.
    """
    print("=" * 80)
    print("COMPARAISON VERSION 2.0 vs VERSION 3.0")
    print("=" * 80)
    print()

    # Phrases de test
    test_cases = [
        ("Le chat mange une souris", "Le félin dévore un rat"),
        ("La voiture roule vite", "L'automobile se déplace rapidement"),
        ("Les enfants jouent dans le jardin", "Les gamins s'amusent au parc"),
        ("Le médecin soigne le patient", "Le docteur traite le malade"),
    ]

    for i, (s1, s2) in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Phrase 1: \"{s1}\"")
        print(f"  Phrase 2: \"{s2}\"")
        print()

        # Version 2.0 (sans les nouvelles fonctionnalités)
        calc_v2 = JaccardSimilarity(
            remove_stopwords=True,
            use_stemming=True
        )
        sim_v2 = calc_v2.calculate_similarity(s1, s2)
        print(f"  VERSION 2.0 (stemming + stop-words):")
        print(f"    Similarité: {sim_v2:.4f}")
        print()

        # Version 3.0 (avec lemmatisation)
        calc_v3_lemma = JaccardSimilarity(
            remove_stopwords=True,
            use_lemmatization=True
        )
        sim_v3_lemma = calc_v3_lemma.calculate_similarity(s1, s2)
        print(f"  VERSION 3.0 (lemmatisation + stop-words):")
        print(f"    Similarité: {sim_v3_lemma:.4f}")
        print()

        # Version 3.0 (avec synonymes)
        calc_v3_syn = JaccardSimilarity(
            remove_stopwords=True,
            use_lemmatization=True,
            use_synonyms=True
        )
        sim_v3_syn = calc_v3_syn.calculate_similarity(s1, s2)
        result_v3_syn = calc_v3_syn.calculate_similarity_detailed(s1, s2)
        print(f"  VERSION 3.0 (lemmatisation + synonymes + stop-words):")
        print(f"    Similarité: {sim_v3_syn:.4f}")
        if 'common_via_synonyms_count' in result_v3_syn:
            print(f"    Mots communs (avec synonymes): {result_v3_syn['common_via_synonyms_count']}")
        print()

        # Version 3.0 COMPLÈTE (avec analyse sémantique)
        calc_v3_full = JaccardSimilarity(
            remove_stopwords=True,
            use_lemmatization=True,
            use_synonyms=True,
            use_semantic_analysis=True
        )
        result_v3_full = calc_v3_full.calculate_similarity_detailed(s1, s2)
        print(f"  VERSION 3.0 COMPLÈTE (lemmatisation + synonymes + sémantique):")
        print(f"    Similarité Jaccard: {result_v3_full['jaccard_similarity']:.4f}")
        if 'semantic_similarity' in result_v3_full:
            print(f"    Similarité sémantique: {result_v3_full['semantic_similarity']:.4f}")
        if 'hybrid_similarity' in result_v3_full:
            print(f"    Similarité hybride: {result_v3_full['hybrid_similarity']:.4f}")

        print("-" * 80)
        print()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Calcul de similarité de Jaccard',
        epilog='Nouvelles fonctionnalités v3.0:\n'
               '  --use-lemmatization: Lemmatisation avancée\n'
               '  --use-synonyms: Gestion des synonymes\n'
               '  --use-semantic: Analyse sémantique\n\n'
               'Exemples d\'utilisation:\n'
               '  python jaccard_similarity.py --interactive\n'
               '  python jaccard_similarity.py --interactive --use-lemmatization\n'
               '  python jaccard_similarity.py --interactive --use-synonyms --use-semantic\n'
               '  python jaccard_similarity.py --demo\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--case-sensitive', action='store_true',
                        help='Respecte la casse des mots')
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='Retire les stop-words français')
    parser.add_argument('--use-stemming', action='store_true',
                        help='Applique le stemming (v2.0)')
    parser.add_argument('--use-lemmatization', action='store_true',
                        help='Applique la lemmatisation avancée')
    parser.add_argument('--use-synonyms', action='store_true',
                        help='Gère les synonymes')
    parser.add_argument('--use-semantic', action='store_true',
                        help='Active l\'analyse sémantique')
    parser.add_argument('--interactive', action='store_true',
                        help='Mode interactif pour saisir des phrases')
    parser.add_argument('--demo', action='store_true',
                        help='Démo de comparaison v2.0 vs v3.0')
    parser.add_argument('--export', choices=['json'],
                        help='Exporte les résultats')

    args = parser.parse_args()

    # Configuration du calculateur avec toutes les options
    calculator = JaccardSimilarity(
        case_sensitive=args.case_sensitive,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        use_lemmatization=args.use_lemmatization,
        use_synonyms=args.use_synonyms,
        use_semantic_analysis=args.use_semantic
    )

    # Mode démo
    if args.demo:
        run_comparison_v2_v3()
        return

    # Mode interactif
    if args.interactive:
        interactive_mode(calculator)
        return

    # Mode par défaut : affichage des informations
    print("=" * 80)
    print("CALCULATEUR DE SIMILARITÉ DE JACCARD")
    print("=" * 80)
    print()
    print(calculator.get_config_summary())
    print()
    print("💡 Modes disponibles:")
    print("  --interactive : Mode interactif pour saisir vos phrases")
    print("  --demo        : Comparaison v2.0 vs v3.0")
    print()
    print("💡 Options v3.0:")
    print("  --use-lemmatization : Lemmatisation avancée")
    print("  --use-synonyms      : Gestion des synonymes")
    print("  --use-semantic      : Analyse sémantique")
    print()
    print("💡 Exemples:")
    print("  python jaccard_similarity.py --interactive")
    print("  python jaccard_similarity.py --interactive --use-lemmatization --use-synonyms")
    print("  python jaccard_similarity.py --demo")
    print()


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()