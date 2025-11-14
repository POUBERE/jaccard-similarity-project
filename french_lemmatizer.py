#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de lemmatisation française avancée

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Novembre 2025
"""

from typing import Dict, Set


class FrenchLemmatizer:
    """Lemmatiseur français avancé."""

    # Dictionnaire complet des verbes irréguliers français
    VERB_LEMMAS = {
        # Être
        'suis': 'être', 'es': 'être', 'est': 'être',
        'sommes': 'être', 'êtes': 'être', 'sont': 'être',
        'étais': 'être', 'était': 'être', 'étions': 'être', 'étiez': 'être', 'étaient': 'être',
        'fus': 'être', 'fut': 'être', 'fûmes': 'être', 'fûtes': 'être', 'furent': 'être',
        'serai': 'être', 'seras': 'être', 'sera': 'être', 'serons': 'être', 'serez': 'être', 'seront': 'être',
        'serais': 'être', 'serait': 'être', 'serions': 'être', 'seriez': 'être', 'seraient': 'être',
        'sois': 'être', 'soit': 'être', 'soyons': 'être', 'soyez': 'être', 'soient': 'être',
        'fusse': 'être', 'fût': 'être', 'fussions': 'être', 'fussiez': 'être', 'fussent': 'être',
        'été': 'être', 'étant': 'être',

        # Avoir
        'ai': 'avoir', 'as': 'avoir', 'a': 'avoir',
        'avons': 'avoir', 'avez': 'avoir', 'ont': 'avoir',
        'avais': 'avoir', 'avait': 'avoir', 'avions': 'avoir', 'aviez': 'avoir', 'avaient': 'avoir',
        'eus': 'avoir', 'eut': 'avoir', 'eûmes': 'avoir', 'eûtes': 'avoir', 'eurent': 'avoir',
        'aurai': 'avoir', 'auras': 'avoir', 'aura': 'avoir', 'aurons': 'avoir', 'aurez': 'avoir', 'auront': 'avoir',
        'aurais': 'avoir', 'aurait': 'avoir', 'aurions': 'avoir', 'auriez': 'avoir', 'auraient': 'avoir',
        'aie': 'avoir', 'aies': 'avoir', 'ait': 'avoir', 'ayons': 'avoir', 'ayez': 'avoir', 'aient': 'avoir',
        'eusse': 'avoir', 'eût': 'avoir', 'eussions': 'avoir', 'eussiez': 'avoir', 'eussent': 'avoir',
        'eu': 'avoir', 'ayant': 'avoir',

        # Aller
        'vais': 'aller', 'vas': 'aller', 'va': 'aller',
        'allons': 'aller', 'allez': 'aller', 'vont': 'aller',
        'allais': 'aller', 'allait': 'aller', 'allions': 'aller', 'alliez': 'aller', 'allaient': 'aller',
        'allai': 'aller', 'alla': 'aller', 'allâmes': 'aller', 'allâtes': 'aller', 'allèrent': 'aller',
        'irai': 'aller', 'iras': 'aller', 'ira': 'aller', 'irons': 'aller', 'irez': 'aller', 'iront': 'aller',
        'irais': 'aller', 'irait': 'aller', 'irions': 'aller', 'iriez': 'aller', 'iraient': 'aller',
        'aille': 'aller', 'ailles': 'aller', 'aillent': 'aller',
        'allasse': 'aller', 'allât': 'aller', 'allassions': 'aller', 'allassiez': 'aller', 'allassent': 'aller',
        'allé': 'aller', 'allée': 'aller', 'allés': 'aller', 'allées': 'aller', 'allant': 'aller',

        # Faire
        'fais': 'faire', 'fait': 'faire', 'faisons': 'faire', 'faites': 'faire', 'font': 'faire',
        'faisais': 'faire', 'faisait': 'faire', 'faisions': 'faire', 'faisiez': 'faire', 'faisaient': 'faire',
        'fis': 'faire', 'fit': 'faire', 'fîmes': 'faire', 'fîtes': 'faire', 'firent': 'faire',
        'ferai': 'faire', 'feras': 'faire', 'fera': 'faire', 'ferons': 'faire', 'ferez': 'faire', 'feront': 'faire',
        'ferais': 'faire', 'ferait': 'faire', 'ferions': 'faire', 'feriez': 'faire', 'feraient': 'faire',
        'fasse': 'faire', 'fasses': 'faire', 'fassent': 'faire', 'fassions': 'faire', 'fassiez': 'faire',
        'faisant': 'faire',

        # Dire
        'dis': 'dire', 'dit': 'dire', 'disons': 'dire', 'dites': 'dire', 'disent': 'dire',
        'disais': 'dire', 'disait': 'dire', 'disions': 'dire', 'disiez': 'dire', 'disaient': 'dire',
        'dirai': 'dire', 'diras': 'dire', 'dira': 'dire', 'dirons': 'dire', 'direz': 'dire', 'diront': 'dire',
        'dirais': 'dire', 'dirait': 'dire', 'dirions': 'dire', 'diriez': 'dire', 'diraient': 'dire',
        'dise': 'dire', 'dises': 'dire', 'disant': 'dire',

        # Pouvoir
        'peux': 'pouvoir', 'peut': 'pouvoir', 'pouvons': 'pouvoir', 'pouvez': 'pouvoir', 'peuvent': 'pouvoir',
        'pouvais': 'pouvoir', 'pouvait': 'pouvoir', 'pouvions': 'pouvoir', 'pouviez': 'pouvoir', 'pouvaient': 'pouvoir',
        'pus': 'pouvoir', 'put': 'pouvoir', 'pûmes': 'pouvoir', 'pûtes': 'pouvoir', 'purent': 'pouvoir',
        'pourrai': 'pouvoir', 'pourras': 'pouvoir', 'pourra': 'pouvoir', 'pourrons': 'pouvoir', 'pourrez': 'pouvoir', 'pourront': 'pouvoir',
        'pourrais': 'pouvoir', 'pourrait': 'pouvoir', 'pourrions': 'pouvoir', 'pourriez': 'pouvoir', 'pourraient': 'pouvoir',
        'puisse': 'pouvoir', 'puisses': 'pouvoir', 'puissions': 'pouvoir', 'puissiez': 'pouvoir', 'puissent': 'pouvoir',
        'pu': 'pouvoir', 'pouvant': 'pouvoir',

        # Vouloir
        'veux': 'vouloir', 'veut': 'vouloir', 'voulons': 'vouloir', 'voulez': 'vouloir', 'veulent': 'vouloir',
        'voulais': 'vouloir', 'voulait': 'vouloir', 'voulions': 'vouloir', 'vouliez': 'vouloir', 'voulaient': 'vouloir',
        'voulus': 'vouloir', 'voulut': 'vouloir', 'voulûmes': 'vouloir', 'voulûtes': 'vouloir', 'voulurent': 'vouloir',
        'voudrai': 'vouloir', 'voudras': 'vouloir', 'voudra': 'vouloir', 'voudrons': 'vouloir', 'voudrez': 'vouloir', 'voudront': 'vouloir',
        'voudrais': 'vouloir', 'voudrait': 'vouloir', 'voudrions': 'vouloir', 'voudriez': 'vouloir', 'voudraient': 'vouloir',
        'veuille': 'vouloir', 'veuilles': 'vouloir', 'veuillent': 'vouloir', 'veuillions': 'vouloir', 'veuillez': 'vouloir',
        'voulu': 'vouloir', 'voulant': 'vouloir',

        # Venir
        'viens': 'venir', 'vient': 'venir', 'venons': 'venir', 'venez': 'venir', 'viennent': 'venir',
        'venais': 'venir', 'venait': 'venir', 'venions': 'venir', 'veniez': 'venir', 'venaient': 'venir',
        'vins': 'venir', 'vint': 'venir', 'vînmes': 'venir', 'vîntes': 'venir', 'vinrent': 'venir',
        'viendrai': 'venir', 'viendras': 'venir', 'viendra': 'venir', 'viendrons': 'venir', 'viendrez': 'venir', 'viendront': 'venir',
        'viendrais': 'venir', 'viendrait': 'venir', 'viendrions': 'venir', 'viendriez': 'venir', 'viendraient': 'venir',
        'vienne': 'venir', 'viennes': 'venir', 'viennent': 'venir', 'venions': 'venir',
        'venu': 'venir', 'venue': 'venir', 'venus': 'venir', 'venues': 'venir', 'venant': 'venir',

        # Voir
        'vois': 'voir', 'voit': 'voir', 'voyons': 'voir', 'voyez': 'voir', 'voient': 'voir',
        'voyais': 'voir', 'voyait': 'voir', 'voyions': 'voir', 'voyiez': 'voir', 'voyaient': 'voir',
        'vis': 'voir', 'vit': 'voir', 'vîmes': 'voir', 'vîtes': 'voir', 'virent': 'voir',
        'verrai': 'voir', 'verras': 'voir', 'verra': 'voir', 'verrons': 'voir', 'verrez': 'voir', 'verront': 'voir',
        'verrais': 'voir', 'verrait': 'voir', 'verrions': 'voir', 'verriez': 'voir', 'verraient': 'voir',
        'voie': 'voir', 'voies': 'voir', 'voient': 'voir', 'voyions': 'voir',
        'vu': 'voir', 'vue': 'voir', 'vus': 'voir', 'vues': 'voir', 'voyant': 'voir',

        # Savoir
        'sais': 'savoir', 'sait': 'savoir', 'savons': 'savoir', 'savez': 'savoir', 'savent': 'savoir',
        'savais': 'savoir', 'savait': 'savoir', 'savions': 'savoir', 'saviez': 'savoir', 'savaient': 'savoir',
        'sus': 'savoir', 'sut': 'savoir', 'sûmes': 'savoir', 'sûtes': 'savoir', 'surent': 'savoir',
        'saurai': 'savoir', 'sauras': 'savoir', 'saura': 'savoir', 'saurons': 'savoir', 'saurez': 'savoir', 'sauront': 'savoir',
        'saurais': 'savoir', 'saurait': 'savoir', 'saurions': 'savoir', 'sauriez': 'savoir', 'sauraient': 'savoir',
        'sache': 'savoir', 'saches': 'savoir', 'sachent': 'savoir', 'sachions': 'savoir', 'sachiez': 'savoir',
        'su': 'savoir', 'sue': 'savoir', 'sus': 'savoir', 'sues': 'savoir', 'sachant': 'savoir',

        # Prendre
        'prends': 'prendre', 'prend': 'prendre', 'prenons': 'prendre', 'prenez': 'prendre', 'prennent': 'prendre',
        'prenais': 'prendre', 'prenait': 'prendre', 'prenions': 'prendre', 'preniez': 'prendre', 'prenaient': 'prendre',
        'pris': 'prendre', 'prit': 'prendre', 'prîmes': 'prendre', 'prîtes': 'prendre', 'prirent': 'prendre',
        'prendrai': 'prendre', 'prendras': 'prendre', 'prendra': 'prendre', 'prendrons': 'prendre', 'prendrez': 'prendre', 'prendront': 'prendre',
        'prendrais': 'prendre', 'prendrait': 'prendre', 'prendrions': 'prendre', 'prendriez': 'prendre', 'prendraient': 'prendre',
        'prenne': 'prendre', 'prennes': 'prendre', 'prennent': 'prendre', 'prenions': 'prendre',
        'prise': 'prendre', 'prises': 'prendre', 'prenant': 'prendre',

        # Mettre
        'mets': 'mettre', 'met': 'mettre', 'mettons': 'mettre', 'mettez': 'mettre', 'mettent': 'mettre',
        'mettais': 'mettre', 'mettait': 'mettre', 'mettions': 'mettre', 'mettiez': 'mettre', 'mettaient': 'mettre',
        'mis': 'mettre', 'mit': 'mettre', 'mîmes': 'mettre', 'mîtes': 'mettre', 'mirent': 'mettre',
        'mettrai': 'mettre', 'mettras': 'mettre', 'mettra': 'mettre', 'mettrons': 'mettre', 'mettrez': 'mettre', 'mettront': 'mettre',
        'mettrais': 'mettre', 'mettrait': 'mettre', 'mettrions': 'mettre', 'mettriez': 'mettre', 'mettraient': 'mettre',
        'mette': 'mettre', 'mettes': 'mettre', 'mettent': 'mettre', 'mettions': 'mettre',
        'mise': 'mettre', 'mises': 'mettre', 'mettant': 'mettre',
        
        # Verbes réguliers en -er courants (AJOUT pour corriger le bug)
        'mange': 'manger', 'manges': 'manger', 'mangent': 'manger',
        'mangé': 'manger', 'mangée': 'manger', 'mangés': 'manger', 'mangées': 'manger',
        'mangeais': 'manger', 'mangeait': 'manger', 'mangions': 'manger', 'mangiez': 'manger', 'mangeaient': 'manger',
        'mangerai': 'manger', 'mangeras': 'manger', 'mangera': 'manger', 'mangerons': 'manger', 'mangerez': 'manger', 'mangeront': 'manger',
        
        'parle': 'parler', 'parles': 'parler', 'parlent': 'parler',
        'parlé': 'parler', 'parlée': 'parler', 'parlés': 'parler', 'parlées': 'parler',
        
        'aime': 'aimer', 'aimes': 'aimer', 'aiment': 'aimer',
        'aimé': 'aimer', 'aimée': 'aimer', 'aimés': 'aimer', 'aimées': 'aimer',
        
        'donne': 'donner', 'donnes': 'donner', 'donnent': 'donner',
        'donné': 'donner', 'donnée': 'donner', 'donnés': 'donner', 'données': 'donner',
        
        'trouve': 'trouver', 'trouves': 'trouver', 'trouvent': 'trouver',
        'trouvé': 'trouver', 'trouvée': 'trouver', 'trouvés': 'trouver', 'trouvées': 'trouver',
        
        'pense': 'penser', 'penses': 'penser', 'pensent': 'penser',
        'pensé': 'penser', 'pensée': 'penser', 'pensés': 'penser', 'pensées': 'penser',
        
        'reste': 'rester', 'restes': 'rester', 'restent': 'rester',
        'resté': 'rester', 'restée': 'rester', 'restés': 'rester', 'restées': 'rester',
        
        'passe': 'passer', 'passes': 'passer', 'passent': 'passer',
        'passé': 'passer', 'passée': 'passer', 'passés': 'passer', 'passées': 'passer',
    }

    # Dictionnaire de noms au pluriel -> singulier (ÉTENDU)
    NOUN_LEMMAS = {
        # Pluriels irréguliers
        'chevaux': 'cheval',
        'travaux': 'travail',
        'baux': 'bail',
        'coraux': 'corail',
        'émaux': 'émail',
        'vitraux': 'vitrail',
        'yeux': 'œil',
        'cieux': 'ciel',
        'aïeux': 'aïeul',

        # Noms en -al
        'animaux': 'animal',
        'journaux': 'journal',
        'hôpitaux': 'hôpital',
        'canaux': 'canal',
        'bocaux': 'bocal',
        'locaux': 'local',
        'capitaux': 'capital',
        'généraux': 'général',

        # Noms en -eau
        'bateaux': 'bateau',
        'châteaux': 'château',
        'couteaux': 'couteau',
        'gâteaux': 'gâteau',
        'drapeaux': 'drapeau',
        'oiseaux': 'oiseau',
        'tableaux': 'tableau',
        'morceaux': 'morceau',
        'niveaux': 'niveau',

        # Noms en -ou
        'bijoux': 'bijou',
        'cailloux': 'caillou',
        'choux': 'chou',
        'genoux': 'genou',
        'hiboux': 'hibou',
        'joujoux': 'joujou',
        'poux': 'pou',

        # Autres pluriels
        'enfants': 'enfant',
        'gens': 'gens',
        'messieurs': 'monsieur',
        'mesdames': 'madame',
        'mesdemoiselles': 'mademoiselle',
        
        # Fruits et légumes
        'mangues': 'mangue',
        'pommes': 'pomme',
        'poires': 'poire',
        'bananes': 'banane',
        'oranges': 'orange',
        'fraises': 'fraise',
        'cerises': 'cerise',
        'prunes': 'prune',
        'pêches': 'pêche',
        'tomates': 'tomate',
        'carottes': 'carotte',
        'salades': 'salade',
        
        # Autres noms courants
        'tables': 'table',
        'chaises': 'chaise',
        'portes': 'porte',
        'fenêtres': 'fenêtre',
        'voitures': 'voiture',
        'maisons': 'maison',
        'personnes': 'personne',
        'choses': 'chose',
        'places': 'place',
        'phrases': 'phrase',
        'pages': 'page',
        'images': 'image',
        'heures': 'heure',
        'minutes': 'minute',
        'semaines': 'semaine',
        'années': 'année',
        'villes': 'ville',
        'routes': 'route',
        'lettres': 'lettre',
        'notes': 'note',
        'fautes': 'faute',
        'dates': 'date',
        'forces': 'force',
        'formes': 'forme',
    }

    # Dictionnaire d'adjectifs féminin -> masculin
    ADJECTIVE_LEMMAS = {
        'belle': 'beau',
        'belles': 'beau',
        'bonne': 'bon',
        'bonnes': 'bon',
        'grande': 'grand',
        'grandes': 'grand',
        'petite': 'petit',
        'petites': 'petit',
        'heureuse': 'heureux',
        'heureuses': 'heureux',
        'blanche': 'blanc',
        'blanches': 'blanc',
        'douce': 'doux',
        'douces': 'doux',
        'fraîche': 'frais',
        'fraîches': 'frais',
        'longue': 'long',
        'longues': 'long',
        'nouvelle': 'nouveau',
        'nouvelles': 'nouveau',
        'vieille': 'vieux',
        'vieilles': 'vieux',
        'ancienne': 'ancien',
        'anciennes': 'ancien',
        'première': 'premier',
        'premières': 'premier',
        'dernière': 'dernier',
        'dernières': 'dernier',
    }

    # Mots protégés (à ne jamais lemmatiser)
    PROTECTED_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'de', 'du', 'au', 'aux', 'ce', 'ces',
        'et', 'ou', 'mais', 'car', 'or', 'donc', 'ni',
        'si', 'ne', 'pas', 'plus', 'très', 'bien', 'tout'
    }

    def __init__(self):
        """Initialise le lemmatiseur."""
        # Fusionner tous les dictionnaires
        self._lemma_dict = {}
        self._lemma_dict.update(self.VERB_LEMMAS)
        self._lemma_dict.update(self.NOUN_LEMMAS)
        self._lemma_dict.update(self.ADJECTIVE_LEMMAS)

    def lemmatize(self, word: str) -> str:
        """
        Lemmatise un mot français.

        Paramètres:
            word (str): Le mot à lemmatiser

        Retourne:
            str: Le lemme (forme canonique) du mot
        """
        word_lower = word.lower()

        # Vérifier si c'est un mot protégé
        if word_lower in self.PROTECTED_WORDS:
            return word_lower

        # Vérifier dans le dictionnaire (priorité absolue)
        if word_lower in self._lemma_dict:
            return self._lemma_dict[word_lower]

        # Les mots trop courts sans lemme connu
        if len(word) <= 2:
            return word_lower

        # Règles de lemmatisation pour les verbes réguliers
        word_lemma = self._lemmatize_regular_verb(word_lower)
        if word_lemma != word_lower:
            return word_lemma

        # Règles pour les pluriels réguliers
        word_lemma = self._lemmatize_plural(word_lower)
        if word_lemma != word_lower:
            return word_lemma

        # Règles pour les adjectifs féminins (EN DERNIER)
        word_lemma = self._lemmatize_feminine(word_lower)
        if word_lemma != word_lower:
            return word_lemma

        return word_lower

    def _lemmatize_regular_verb(self, word: str) -> str:
        """Lemmatise les verbes réguliers."""
        # Verbes en -er (SAUF 'e' et 'es' seuls qui sont gérés dans le dictionnaire)
        if len(word) > 3:
            er_endings = [
                'ons', 'ez', 'ent', 
                'ais', 'ait', 'ions', 'iez', 'aient',
                'ai', 'as', 'a', 'âmes', 'âtes', 'èrent',
                'erai', 'eras', 'era', 'erons', 'erez', 'eront',
                'erais', 'erait', 'erions', 'eriez', 'eraient',
                'ant'
            ]

            for ending in er_endings:
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    stem = word[:-len(ending)]
                    if stem.endswith('e'):
                        return stem + 'r'
                    return stem + 'er'

        # Verbes en -ir
        if len(word) > 3:
            ir_endings = ['is', 'it', 'issons', 'issez', 'issent',
                          'issais', 'issait', 'issions', 'issiez', 'issaient',
                          'irai', 'iras', 'ira', 'irons', 'irez', 'iront']

            for ending in ir_endings:
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    stem = word[:-len(ending)]
                    return stem + 'ir'

        return word

    def _lemmatize_plural(self, word: str) -> str:
        """Lemmatise les pluriels réguliers."""
        if word.endswith('aux'):
            return word[:-3] + 'al'
        elif word.endswith('eaux'):
            return word[:-1]
        elif word.endswith('eux') and len(word) > 4:
            return word[:-1]
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]

        return word

    def _lemmatize_feminine(self, word: str) -> str:
        """Lemmatise les formes féminines - SEULEMENT les adjectifs reconnus."""
        # Liste restreinte d'adjectifs féminins en -e
        # Pour éviter de confondre avec les verbes
        feminine_adjectives = {
            'grande', 'petite', 'forte', 'faible', 'jeune', 'vieille',
            'haute', 'basse', 'courte', 'longue', 'large', 'étroite',
            'rouge', 'verte', 'bleue', 'jaune', 'noire', 'blanche',
            'proche', 'lointaine', 'proche', 'dure', 'molle'
        }
        
        if word.endswith('euse'):
            return word[:-4] + 'eux'
        elif word.endswith('ive'):
            return word[:-3] + 'if'
        elif word.endswith('elle'):
            if word in ['belle', 'nouvelle', 'vieille']:
                return self.ADJECTIVE_LEMMAS.get(word, word[:-2])
            return word[:-2]
        elif word.endswith('enne'):
            return word[:-2]
        elif word.endswith('ière'):
            return word[:-3] + 'ier'
        elif word.endswith('ée') and len(word) > 3:
            return word[:-1]
        # MODIFICATION CRITIQUE: ne traiter 'e' final que pour adjectifs connus
        elif word in feminine_adjectives:
            return word[:-1]

        return word

    def add_custom_lemma(self, word: str, lemma: str) -> None:
        """Ajoute une règle de lemmatisation personnalisée."""
        self._lemma_dict[word.lower()] = lemma.lower()

    def get_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur le lemmatiseur."""
        return {
            'total_lemmas': len(self._lemma_dict),
            'verb_lemmas': len(self.VERB_LEMMAS),
            'noun_lemmas': len(self.NOUN_LEMMAS),
            'adjective_lemmas': len(self.ADJECTIVE_LEMMAS)
        }


if __name__ == "__main__":
    lemmatizer = FrenchLemmatizer()

    print("=== Lemmatiseur Français v3.0.2 - CORRIGÉ ===\n")

    print("🐛 Test du bug critique:")
    critical_test = [
        ('mange', 'manger'),
        ('mangent', 'manger'),
        ('mangue', 'mangue'),
        ('mangues', 'mangue'),
    ]
    for word, expected in critical_test:
        result = lemmatizer.lemmatize(word)
        status = "✓" if result == expected else "❌"
        print(f"  {status} {word:15} -> {result:15} (attendu: {expected})")

    print("\nVerbes réguliers en -er:")
    test_verbs = ['parle', 'parles', 'parlent', 'parlons', 'parlez', 'parlé']
    for verb in test_verbs:
        lemma = lemmatizer.lemmatize(verb)
        print(f"  {verb:15} -> {lemma}")

    print("\nStatistiques:")
    stats = lemmatizer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")