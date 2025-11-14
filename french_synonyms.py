#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de gestion des synonymes français

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Novembre 2025
"""

from typing import Set, Dict, List


class FrenchSynonyms:
    """
    Classe pour gérer les synonymes en français.

    Cette classe fournit un dictionnaire de synonymes courants
    et permet d'étendre les ensembles de mots avec leurs synonymes.
    """

    # Dictionnaire de synonymes français
    # Chaque groupe contient des mots considérés comme synonymes
    SYNONYM_GROUPS = [
        # Animaux
        {'chat', 'félin', 'minet', 'matou'},
        {'chien', 'canin', 'cabot', 'toutou'},
        {'oiseau', 'volatile', 'passereau'},
        {'poisson', 'poiscaille'},

        # Véhicules
        {'voiture', 'automobile', 'auto', 'véhicule', 'bagnole'},
        {'vélo', 'bicyclette', 'cycle'},
        {'avion', 'aéronef', 'appareil'},
        {'bateau', 'navire', 'embarcation', 'vaisseau'},

        # Habitation
        {'maison', 'habitation', 'demeure', 'résidence', 'logement', 'domicile'},
        {'appartement', 'logement', 'studio'},
        {'chambre', 'pièce'},

        # Personnes
        {'enfant', 'gosse', 'gamin', 'bambin', 'môme'},
        {'personne', 'individu', 'être'},
        {'homme', 'monsieur', 'mâle'},
        {'femme', 'dame', 'madame'},
        {'ami', 'copain', 'camarade', 'pote'},
        {'professeur', 'enseignant', 'prof', 'maître', 'instituteur'},
        {'médecin', 'docteur', 'doc', 'toubib'},
        {'élève', 'étudiant', 'apprenant'},

        # Travail
        {'travail', 'emploi', 'job', 'boulot', 'tâche'},
        {'entreprise', 'société', 'compagnie', 'firme'},
        {'patron', 'chef', 'directeur', 'boss'},
        {'salaire', 'rémunération', 'paie', 'paye'},

        # Émotions
        {'content', 'heureux', 'joyeux', 'ravi', 'enchanté'},
        {'triste', 'malheureux', 'chagriné', 'affligé'},
        {'peur', 'crainte', 'effroi', 'frayeur', 'angoisse'},
        {'colère', 'rage', 'fureur', 'courroux'},
        {'amour', 'affection', 'tendresse'},

        # Actions
        {'manger', 'dévorer', 'grignoter', 'avaler', 'consommer'},
        {'boire', 'siroter', 'absorber'},
        {'marcher', 'aller', 'se déplacer', 'cheminer'},
        {'courir', 'sprinter', 'filer'},
        {'parler', 'dire', 'discuter', 'bavarder', 'causer'},
        {'regarder', 'observer', 'voir', 'contempler'},
        {'écouter', 'entendre', 'ouïr'},
        {'dormir', 'sommeiller', 'roupiller'},
        {'travailler', 'bosser', 'œuvrer', 'labeur'},
        {'jouer', 'samuser', 'divertir'},
        {'créer', 'fabriquer', 'produire', 'réaliser'},
        {'détruire', 'démolir', 'casser', 'briser'},
        {'commencer', 'débuter', 'entamer', 'amorcer'},
        {'finir', 'terminer', 'achever', 'conclure'},
        {'donner', 'offrir', 'céder'},
        {'prendre', 'saisir', 'attraper'},

        # Qualités
        {'beau', 'joli', 'magnifique', 'superbe', 'splendide'},
        {'laid', 'vilain', 'hideux', 'moche'},
        {'grand', 'haut', 'élevé', 'immense'},
        {'petit', 'minuscule', 'réduit', 'menu'},
        {'gros', 'volumineux', 'imposant', 'massif'},
        {'mince', 'fin', 'svelte', 'élancé'},
        {'rapide', 'vite', 'prompt', 'véloce'},
        {'lent', 'lentement', 'tranquille'},
        {'bon', 'excellent', 'parfait', 'super'},
        {'mauvais', 'médiocre', 'piètre', 'nul'},
        {'facile', 'simple', 'aisé'},
        {'difficile', 'compliqué', 'ardu', 'complexe'},
        {'chaud', 'brûlant', 'torride'},
        {'froid', 'glacé', 'glacial', 'gelé'},

        # Quantité
        {'beaucoup', 'nombreux', 'abondant', 'multiple'},
        {'peu', 'rare', 'faible'},

        # Temps
        {'maintenant', 'actuellement', 'présentement'},
        {'avant', 'auparavant', 'antérieurement'},
        {'après', 'ensuite', 'puis', 'ultérieurement'},
        {'toujours', 'constamment', 'continuellement'},
        {'jamais', 'nullement'},

        # Lieux
        {'ville', 'cité', 'agglomération'},
        {'campagne', 'province'},
        {'rue', 'avenue', 'boulevard', 'voie'},
        {'magasin', 'boutique', 'commerce'},
        {'école', 'établissement', 'collège', 'lycée'},
        {'hôpital', 'clinique'},

        # Informatique & Technologie
        {'ordinateur', 'pc', 'machine', 'computer'},
        {'téléphone', 'mobile', 'portable', 'smartphone'},
        {'internet', 'web', 'toile', 'net'},
        {'programme', 'logiciel', 'application', 'app'},
        {'code', 'programmation', 'développement'},
        {'données', 'data', 'informations'},

        # Nourriture
        {'pain', 'baguette'},
        {'viande', 'chair'},
        {'légume', 'verdure'},
        {'fruit', 'agrume'},
        {'repas', 'déjeuner', 'dîner', 'souper'},

        # Concepts abstraits
        {'idée', 'concept', 'notion', 'pensée'},
        {'problème', 'difficulté', 'souci', 'ennui'},
        {'solution', 'résolution', 'réponse'},
        {'question', 'interrogation', 'demande'},
        {'réponse', 'réplique', 'riposte'},
        {'importance', 'valeur', 'poids', 'portée'},
        {'intelligence', 'esprit', 'raison'},
        {'connaissance', 'savoir', 'science'},
        {'erreur', 'faute', 'bévue', 'méprise'},
        {'vérité', 'véracité', 'exactitude'},
        {'mensonge', 'fausseté', 'tromperie'},
    ]

    def __init__(self):
        """Initialise le gestionnaire de synonymes."""
        # Créer un dictionnaire pour accès rapide
        # Chaque mot pointe vers son groupe de synonymes
        self._synonym_map: Dict[str, Set[str]] = {}

        for group in self.SYNONYM_GROUPS:
            # Pour chaque mot du groupe, associer tous les autres mots du groupe
            for word in group:
                self._synonym_map[word] = group.copy()

    def get_synonyms(self, word: str) -> Set[str]:
        """
        Retourne l'ensemble des synonymes d'un mot.

        Paramètres:
            word (str): Le mot dont on cherche les synonymes

        Retourne:
            Set[str]: Ensemble des synonymes (inclut le mot lui-même)
        """
        word_lower = word.lower()

        if word_lower in self._synonym_map:
            return self._synonym_map[word_lower].copy()
        else:
            # Si pas de synonyme trouvé, retourner juste le mot
            return {word_lower}

    def expand_with_synonyms(self, words: Set[str]) -> Set[str]:
        """
        Étend un ensemble de mots avec tous leurs synonymes.

        Paramètres:
            words (Set[str]): Ensemble de mots

        Retourne:
            Set[str]: Ensemble étendu avec les synonymes

        Exemple:
            {'chat', 'noir'} -> {'chat', 'félin', 'minet', 'matou', 'noir'}
        """
        expanded = set()

        for word in words:
            # Ajouter le mot et tous ses synonymes
            expanded.update(self.get_synonyms(word))

        return expanded

    def are_synonyms(self, word1: str, word2: str) -> bool:
        """
        Vérifie si deux mots sont synonymes.

        Paramètres:
            word1 (str): Premier mot
            word2 (str): Deuxième mot

        Retourne:
            bool: True si les mots sont synonymes
        """
        word1_lower = word1.lower()
        word2_lower = word2.lower()

        # Deux mots sont synonymes s'ils sont dans le même groupe
        if word1_lower in self._synonym_map:
            return word2_lower in self._synonym_map[word1_lower]

        return word1_lower == word2_lower

    def get_common_synonyms(self, words1: Set[str], words2: Set[str]) -> Set[str]:
        """
        Trouve les mots communs en tenant compte des synonymes.

        Cette méthode considère deux mots comme "communs" s'ils sont
        synonymes, même s'ils ne sont pas identiques.

        Paramètres:
            words1 (Set[str]): Premier ensemble de mots
            words2 (Set[str]): Deuxième ensemble de mots

        Retourne:
            Set[str]: Mots communs (avec synonymes)

        Exemple:
            {'chat'} et {'félin'} -> {'chat', 'félin'}
        """
        # Étendre les deux ensembles avec les synonymes
        expanded1 = self.expand_with_synonyms(words1)
        expanded2 = self.expand_with_synonyms(words2)

        # Intersection des ensembles étendus
        return expanded1.intersection(expanded2)

    def add_custom_synonyms(self, synonym_group: Set[str]) -> None:
        """
        Ajoute un groupe de synonymes personnalisé.

        Paramètres:
            synonym_group (Set[str]): Ensemble de mots synonymes

        Exemple:
            synonyms.add_custom_synonyms({'ia', 'intelligence artificielle', 'ai'})
        """
        # Convertir en minuscules
        group = {word.lower() for word in synonym_group}

        # Ajouter au dictionnaire
        for word in group:
            if word in self._synonym_map:
                # Fusionner avec le groupe existant
                self._synonym_map[word].update(group)
            else:
                # Créer un nouveau groupe
                self._synonym_map[word] = group.copy()

        # Mettre à jour tous les mots du groupe
        for word in group:
            self._synonym_map[word] = self._synonym_map[word].copy()

    def get_stats(self) -> Dict[str, int]:
        """
        Retourne des statistiques sur le dictionnaire de synonymes.

        Retourne:
            Dict: Statistiques (nombre de mots, groupes, etc.)
        """
        unique_groups = []
        seen_groups = set()

        for group in self._synonym_map.values():
            group_key = frozenset(group)
            if group_key not in seen_groups:
                unique_groups.append(group)
                seen_groups.add(group_key)

        return {
            'total_words': len(self._synonym_map),
            'total_groups': len(unique_groups),
            'avg_synonyms_per_word': sum(len(g) for g in unique_groups) / len(unique_groups) if unique_groups else 0
        }


# Exemple d'utilisation
if __name__ == "__main__":
    synonyms = FrenchSynonyms()

    print("=== Gestionnaire de Synonymes Français ===\n")

    # Test 1: Trouver les synonymes d'un mot
    word = "chat"
    syns = synonyms.get_synonyms(word)
    print(f"Synonymes de '{word}': {sorted(syns)}\n")

    # Test 2: Vérifier si deux mots sont synonymes
    w1, w2 = "voiture", "automobile"
    print(f"'{w1}' et '{w2}' sont synonymes: {synonyms.are_synonyms(w1, w2)}\n")

    # Test 3: Étendre un ensemble avec synonymes
    words = {'chat', 'voiture'}
    expanded = synonyms.expand_with_synonyms(words)
    print(f"Ensemble original: {words}")
    print(f"Étendu avec synonymes: {sorted(expanded)}\n")

    # Test 4: Mots communs avec synonymes
    set1 = {'chat', 'noir'}
    set2 = {'félin', 'blanc'}
    common = synonyms.get_common_synonyms(set1, set2)
    print(f"Ensemble 1: {set1}")
    print(f"Ensemble 2: {set2}")
    print(f"Mots communs (avec synonymes): {sorted(common)}\n")

    # Test 5: Statistiques
    stats = synonyms.get_stats()
    print(f"Statistiques du dictionnaire:")
    print(f"  - Mots totaux: {stats['total_words']}")
    print(f"  - Groupes de synonymes: {stats['total_groups']}")
    print(f"  - Moyenne de synonymes par mot: {stats['avg_synonyms_per_word']:.1f}")
