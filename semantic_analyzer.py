#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'analyse sémantique française

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Novembre 2025

Ce module fournit une analyse sémantique basique sans dépendances externes.
Il utilise des champs sémantiques et des relations de similarité contextuelle.
"""

from typing import Dict, Set, List, Tuple
from collections import defaultdict
import math


class SemanticAnalyzer:
    """
    Analyseur sémantique pour le français.

    Cette classe organise les mots en champs sémantiques et permet
    de calculer des similarités sémantiques basiques.
    """

    # Champs sémantiques : groupes de mots liés par le sens
    SEMANTIC_FIELDS = {
        'animaux': {
            'chat', 'chien', 'oiseau', 'poisson', 'cheval', 'vache', 'mouton',
            'poule', 'canard', 'lapin', 'souris', 'rat', 'lion', 'tigre',
            'éléphant', 'singe', 'ours', 'loup', 'renard', 'serpent',
            'félin', 'canin', 'volatile', 'mammifère', 'reptile'
        },

        'véhicules': {
            'voiture', 'vélo', 'moto', 'bus', 'train', 'avion', 'bateau',
            'camion', 'automobile', 'bicyclette', 'motocyclette',
            'navire', 'aéronef', 'transport', 'véhicule'
        },

        'habitation': {
            'maison', 'appartement', 'immeuble', 'habitation', 'logement',
            'résidence', 'domicile', 'demeure', 'chambre', 'pièce', 'salon',
            'cuisine', 'salle', 'garage', 'jardin', 'balcon', 'toit'
        },

        'famille': {
            'père', 'mère', 'fils', 'fille', 'frère', 'sœur', 'oncle', 'tante',
            'cousin', 'cousine', 'grand-père', 'grand-mère', 'enfant', 'parent',
            'famille', 'neveu', 'nièce', 'papa', 'maman'
        },

        'nature': {
            'arbre', 'fleur', 'plante', 'herbe', 'feuille', 'branche', 'racine',
            'forêt', 'bois', 'prairie', 'champ', 'montagne', 'colline', 'vallée',
            'rivière', 'fleuve', 'lac', 'mer', 'océan', 'plage', 'rocher', 'pierre'
        },

        'météo': {
            'pluie', 'neige', 'soleil', 'vent', 'orage', 'tempête', 'nuage',
            'brouillard', 'gel', 'chaleur', 'froid', 'température', 'climat',
            'météo', 'temps', 'saison'
        },

        'nourriture': {
            'pain', 'viande', 'poisson', 'légume', 'fruit', 'fromage', 'lait',
            'œuf', 'riz', 'pâtes', 'salade', 'soupe', 'gâteau', 'dessert',
            'repas', 'déjeuner', 'dîner', 'petit-déjeuner', 'nourriture',
            'aliment', 'cuisine', 'restaurant'
        },

        'corps': {
            'tête', 'bras', 'jambe', 'main', 'pied', 'doigt', 'œil', 'oreille',
            'nez', 'bouche', 'dent', 'cheveux', 'cœur', 'sang', 'os',
            'peau', 'corps', 'visage', 'dos', 'ventre'
        },

        'émotions': {
            'joie', 'tristesse', 'colère', 'peur', 'surprise', 'dégoût',
            'amour', 'haine', 'bonheur', 'malheur', 'plaisir', 'douleur',
            'content', 'heureux', 'triste', 'fâché', 'joyeux', 'émotion',
            'sentiment', 'passion'
        },

        'travail': {
            'travail', 'emploi', 'métier', 'profession', 'job', 'bureau',
            'entreprise', 'société', 'patron', 'employé', 'collègue',
            'salaire', 'argent', 'projet', 'réunion', 'tâche'
        },

        'éducation': {
            'école', 'collège', 'lycée', 'université', 'classe', 'cours',
            'professeur', 'enseignant', 'élève', 'étudiant', 'livre', 'cahier',
            'stylo', 'examen', 'devoir', 'note', 'diplôme', 'éducation',
            'apprentissage', 'leçon'
        },

        'technologie': {
            'ordinateur', 'téléphone', 'internet', 'web', 'logiciel', 'application',
            'programme', 'code', 'données', 'fichier', 'écran', 'clavier',
            'souris', 'réseau', 'serveur', 'cloud', 'email', 'technologie',
            'numérique', 'digital', 'informatique'
        },

        'sport': {
            'football', 'tennis', 'basketball', 'natation', 'course', 'vélo',
            'sport', 'jeu', 'match', 'équipe', 'joueur', 'entraînement',
            'compétition', 'stade', 'ballon', 'victoire', 'défaite'
        },

        'santé': {
            'santé', 'maladie', 'médecin', 'docteur', 'hôpital', 'clinique',
            'médicament', 'traitement', 'soins', 'patient', 'infirmier',
            'chirurgie', 'douleur', 'symptôme', 'diagnostic', 'guérison'
        },

        'temps': {
            'heure', 'minute', 'seconde', 'jour', 'semaine', 'mois', 'année',
            'matin', 'midi', 'après-midi', 'soir', 'nuit', 'aujourd\'hui',
            'hier', 'demain', 'maintenant', 'temps', 'durée', 'période'
        },

        'couleurs': {
            'rouge', 'bleu', 'vert', 'jaune', 'orange', 'violet', 'rose',
            'blanc', 'noir', 'gris', 'marron', 'beige', 'couleur', 'teinte'
        },

        'vêtements': {
            'pantalon', 'chemise', 'robe', 'jupe', 'veste', 'manteau', 'pull',
            'chaussure', 'chaussette', 'chapeau', 'écharpe', 'gant',
            'vêtement', 'habit', 'costume', 'tenue'
        }
    }

    # Relations négatives (mots antonymiques)
    ANTONYMS = {
        ('grand', 'petit'),
        ('chaud', 'froid'),
        ('haut', 'bas'),
        ('bon', 'mauvais'),
        ('beau', 'laid'),
        ('rapide', 'lent'),
        ('fort', 'faible'),
        ('riche', 'pauvre'),
        ('jeune', 'vieux'),
        ('nouveau', 'ancien'),
        ('heureux', 'triste'),
        ('jour', 'nuit'),
        ('entrée', 'sortie'),
        ('début', 'fin'),
        ('avant', 'après'),
        ('facile', 'difficile'),
        ('simple', 'complexe'),
        ('clair', 'obscur'),
        ('ouvert', 'fermé'),
        ('plein', 'vide'),
    }

    def __init__(self):
        """Initialise l'analyseur sémantique."""
        # Créer un index inverse : mot -> champs sémantiques
        self._word_to_fields: Dict[str, Set[str]] = defaultdict(set)

        for field_name, words in self.SEMANTIC_FIELDS.items():
            for word in words:
                self._word_to_fields[word].add(field_name)

        # Convertir les antonymes en dictionnaire
        self._antonyms_dict: Dict[str, Set[str]] = defaultdict(set)
        for word1, word2 in self.ANTONYMS:
            self._antonyms_dict[word1].add(word2)
            self._antonyms_dict[word2].add(word1)

    def get_semantic_fields(self, word: str) -> Set[str]:
        """
        Retourne les champs sémantiques auxquels appartient un mot.

        Paramètres:
            word (str): Le mot à analyser

        Retourne:
            Set[str]: Ensemble des champs sémantiques
        """
        return self._word_to_fields.get(word.lower(), set())

    def are_semantically_related(self, word1: str, word2: str) -> bool:
        """
        Vérifie si deux mots sont liés sémantiquement.

        Deux mots sont liés s'ils appartiennent au même champ sémantique.

        Paramètres:
            word1 (str): Premier mot
            word2 (str): Deuxième mot

        Retourne:
            bool: True si les mots sont liés
        """
        fields1 = self.get_semantic_fields(word1)
        fields2 = self.get_semantic_fields(word2)

        # Vérifier s'il y a intersection des champs
        return bool(fields1.intersection(fields2))

    def semantic_similarity(self, word1: str, word2: str) -> float:
        """
        Calcule une similarité sémantique entre deux mots.

        La similarité est basée sur le nombre de champs sémantiques partagés.

        Paramètres:
            word1 (str): Premier mot
            word2 (str): Deuxième mot

        Retourne:
            float: Score de similarité entre 0 et 1
        """
        # Même mot = similarité maximale
        if word1.lower() == word2.lower():
            return 1.0

        # Vérifier les antonymes
        if word2.lower() in self._antonyms_dict.get(word1.lower(), set()):
            return 0.0

        fields1 = self.get_semantic_fields(word1)
        fields2 = self.get_semantic_fields(word2)

        if not fields1 or not fields2:
            return 0.0

        # Similarité = Jaccard sur les champs sémantiques
        intersection = fields1.intersection(fields2)
        union = fields1.union(fields2)

        return len(intersection) / len(union) if union else 0.0

    def semantic_distance(self, word1: str, word2: str) -> float:
        """
        Calcule une distance sémantique entre deux mots.

        Paramètres:
            word1 (str): Premier mot
            word2 (str): Deuxième mot

        Retourne:
            float: Distance entre 0 et 1
        """
        return 1.0 - self.semantic_similarity(word1, word2)

    def expand_with_semantic_context(self, words: Set[str]) -> Set[str]:
        """
        Étend un ensemble de mots avec leur contexte sémantique.

        Pour chaque mot, ajoute quelques mots du même champ sémantique.

        Paramètres:
            words (Set[str]): Ensemble de mots

        Retourne:
            Set[str]: Ensemble étendu
        """
        expanded = set(words)

        for word in words:
            fields = self.get_semantic_fields(word)

            # Ajouter quelques mots de chaque champ (max 3 par champ)
            for field in fields:
                field_words = self.SEMANTIC_FIELDS[field]
                # Prendre les 3 premiers mots du champ (qui ne sont pas déjà dans words)
                count = 0
                for fw in field_words:
                    if fw not in words and count < 3:
                        expanded.add(fw)
                        count += 1

        return expanded

    def semantic_sentence_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        """
        Calcule une similarité sémantique entre deux ensembles de mots.

        Cette méthode prend en compte les relations sémantiques entre mots,
        pas seulement les correspondances exactes.

        Paramètres:
            words1 (Set[str]): Premier ensemble
            words2 (Set[str]): Deuxième ensemble

        Retourne:
            float: Score de similarité entre 0 et 1
        """
        if not words1 or not words2:
            return 0.0

        # Calculer une matrice de similarité
        total_similarity = 0.0
        comparisons = 0

        for w1 in words1:
            for w2 in words2:
                total_similarity += self.semantic_similarity(w1, w2)
                comparisons += 1

        if comparisons == 0:
            return 0.0

        # Moyenne de similarité
        avg_similarity = total_similarity / comparisons

        return avg_similarity

    def get_related_words(self, word: str, max_words: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les mots les plus liés sémantiquement à un mot donné.

        Paramètres:
            word (str): Le mot de référence
            max_words (int): Nombre maximum de mots à retourner

        Retourne:
            List[Tuple[str, float]]: Liste de (mot, score) triée par score décroissant
        """
        fields = self.get_semantic_fields(word)

        if not fields:
            return []

        # Collecter tous les mots des champs sémantiques
        related = []
        for field in fields:
            for fw in self.SEMANTIC_FIELDS[field]:
                if fw != word.lower():
                    score = self.semantic_similarity(word, fw)
                    if score > 0:
                        related.append((fw, score))

        # Trier par score décroissant
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:max_words]

    def add_semantic_field(self, field_name: str, words: Set[str]) -> None:
        """
        Ajoute un nouveau champ sémantique personnalisé.

        Paramètres:
            field_name (str): Nom du champ
            words (Set[str]): Ensemble de mots du champ
        """
        # Convertir en minuscules
        words_lower = {w.lower() for w in words}

        # Ajouter au dictionnaire principal
        self.SEMANTIC_FIELDS[field_name] = words_lower

        # Mettre à jour l'index inverse
        for word in words_lower:
            self._word_to_fields[word].add(field_name)

    def get_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur l'analyseur."""
        total_words = sum(len(words) for words in self.SEMANTIC_FIELDS.values())

        return {
            'total_fields': len(self.SEMANTIC_FIELDS),
            'total_words': total_words,
            'total_antonyms': len(self.ANTONYMS),
            'avg_words_per_field': total_words // len(self.SEMANTIC_FIELDS) if self.SEMANTIC_FIELDS else 0
        }


# Exemple d'utilisation
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()

    print("=== Analyseur Sémantique Français ===\n")

    # Test 1: Champs sémantiques
    word = "chat"
    fields = analyzer.get_semantic_fields(word)
    print(f"Champs sémantiques de '{word}': {fields}\n")

    # Test 2: Mots liés
    print(f"Mots sémantiquement liés à '{word}':")
    related = analyzer.get_related_words(word, max_words=5)
    for w, score in related:
        print(f"  {w:15} (score: {score:.2f})")
    print()

    # Test 3: Similarité sémantique
    pairs = [
        ("chat", "chien"),
        ("chat", "félin"),
        ("voiture", "vélo"),
        ("pain", "école"),
        ("grand", "petit")
    ]

    print("Similarités sémantiques:")
    for w1, w2 in pairs:
        sim = analyzer.semantic_similarity(w1, w2)
        print(f"  {w1:10} <-> {w2:10} : {sim:.2f}")
    print()

    # Test 4: Analyse de phrases
    set1 = {"chat", "mange", "souris"}
    set2 = {"chien", "dévore", "rat"}

    sem_sim = analyzer.semantic_sentence_similarity(set1, set2)
    print(f"Ensemble 1: {set1}")
    print(f"Ensemble 2: {set2}")
    print(f"Similarité sémantique: {sem_sim:.2f}\n")

    # Test 5: Statistiques
    stats = analyzer.get_stats()
    print("Statistiques:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
