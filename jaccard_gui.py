#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique COMPLÈTE pour le calculateur de similarité de Jaccard
Version 2.2 - TOUTES LES FONCTIONNALITÉS

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

Implémente TOUTES les fonctionnalités de jaccard_similarity.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict
import sys
import os
from datetime import datetime

from jaccard_similarity import JaccardSimilarity, FrenchStemmer


class JaccardGUI:
    """Interface graphique complète avec TOUTES les fonctionnalités."""

    def __init__(self, root):
        """Initialise l'interface graphique."""
        self.root = root
        self.root.title("Calculateur de Jaccard v2.2 - Interface Complète")

        self.root.geometry("1300x950")
        self.root.minsize(1000, 750)

        # Configuration responsive
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=0)
        self.root.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'light': '#ECF0F1',
            'dark': '#34495E',
            'purple': '#9B59B6'
        }

        self.calculator = JaccardSimilarity()

        # Variables pour les options
        self.case_sensitive = tk.BooleanVar(value=False)
        self.remove_punctuation = tk.BooleanVar(value=True)
        self.remove_stopwords = tk.BooleanVar(value=False)
        self.use_stemming = tk.BooleanVar(value=False)
        self.context_var = tk.StringVar(value='general')

        self.phrases_list = []
        self.create_widgets()

    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        self.create_header()
        self.create_options_frame()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)

        # Tous les onglets avec TOUTES les fonctionnalités
        self.create_simple_comparison_tab()
        self.create_multiple_comparison_tab()
        self.create_matrix_tab()
        self.create_extreme_pairs_tab()  # NOUVEAU
        self.create_demo_tests_tab()  # NOUVEAU - Tests automatiques
        self.create_export_tab()
        self.create_about_tab()

        self.create_status_bar()

    def create_header(self):
        """Crée l'en-tête."""
        header_frame = tk.Frame(
            self.root, bg=self.colors['primary'], height=80)
        header_frame.grid(row=0, column=0, sticky='ew')
        header_frame.grid_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="📊 Calculateur de Jaccard v2.2 - Interface Complète",
            font=('Arial', 20, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="TOUTES les fonctionnalités de jaccard_similarity.py",
            font=('Arial', 10),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        subtitle_label.pack()

    def create_options_frame(self):
        """Crée le cadre des options."""
        options_frame = tk.LabelFrame(
            self.root,
            text="⚙️ Configuration Complète",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        options_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        row1 = tk.Frame(options_frame)
        row1.pack(fill='x', pady=2)

        tk.Checkbutton(
            row1, text="Sensible à la casse",
            variable=self.case_sensitive,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Supprimer ponctuation",
            variable=self.remove_punctuation,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Retirer stop-words",
            variable=self.remove_stopwords,
            command=self.update_calculator
        ).pack(side='left', padx=15)

        tk.Checkbutton(
            row1, text="Stemming français",
            variable=self.use_stemming,
            command=self.update_calculator
        ).pack(side='left', padx=15)

    def update_calculator(self):
        """Met à jour le calculateur."""
        self.calculator = JaccardSimilarity(
            case_sensitive=self.case_sensitive.get(),
            remove_punctuation=self.remove_punctuation.get(),
            remove_stopwords=self.remove_stopwords.get(),
            use_stemming=self.use_stemming.get()
        )
        self.update_status("Configuration mise à jour")

    def create_simple_comparison_tab(self):
        """Onglet de comparaison simple COMPLET."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Simple  ")

        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        scrollbar.grid(row=0, column=1, sticky='ns')

        scrollable_frame.columnconfigure(0, weight=1)

        # Phrase 1
        tk.Label(scrollable_frame, text="Phrase 1:", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase1_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase1_text.grid(row=1, column=0, sticky='ew', pady=5, padx=10)

        # Phrase 2
        tk.Label(scrollable_frame, text="Phrase 2:", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky='w', pady=(10, 5), padx=10)

        self.phrase2_text = scrolledtext.ScrolledText(
            scrollable_frame, height=4, font=('Arial', 10), wrap=tk.WORD)
        self.phrase2_text.grid(row=3, column=0, sticky='ew', pady=5, padx=10)

        # Contexte
        context_frame = tk.LabelFrame(
            scrollable_frame,
            text="🎯 Contexte d'interprétation",
            font=('Arial', 10, 'bold'),
            padx=10, pady=5
        )
        context_frame.grid(row=4, column=0, sticky='ew', pady=10, padx=10)

        contexts = [
            ('Général', 'general'),
            ('Plagiat', 'plagiarism'),
            ('Clustering', 'clustering'),
            ('Recherche', 'search'),
            ('Diversité', 'diversity')
        ]

        context_inner = tk.Frame(context_frame)
        context_inner.pack(fill='x', expand=True)

        for label, value in contexts:
            tk.Radiobutton(
                context_inner, text=label,
                variable=self.context_var, value=value
            ).pack(side='left', padx=10, expand=True)

        # Boutons
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=5, column=0, pady=15, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        tk.Button(
            button_frame, text="🔍 Analyse Complète",
            command=self.calculate_complete_analysis,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=0, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="📊 Détails Techniques",
            command=self.show_technical_details,
            bg=self.colors['purple'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=1, padx=3, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Effacer",
            command=self.clear_simple_comparison,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8, cursor='hand2'
        ).grid(row=0, column=2, padx=3, sticky='ew')

        # Résultats
        result_frame = tk.LabelFrame(
            scrollable_frame, text="📊 Résultats Complets",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=6, column=0, sticky='ew', pady=10, padx=10)
        result_frame.columnconfigure(0, weight=1)

        self.simple_result_text = scrolledtext.ScrolledText(
            result_frame, height=20, font=('Courier', 9),
            wrap=tk.WORD, state='disabled'
        )
        self.simple_result_text.pack(fill='both', expand=True)

    def calculate_complete_analysis(self):
        """Analyse COMPLÈTE avec toutes les métriques."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Calcul détaillé complet
            result = self.calculator.calculate_distance_detailed(
                phrase1, phrase2)
            similarity = result['jaccard_similarity']
            distance = result['jaccard_distance']

            context = self.context_var.get()
            sim_interp = self.calculator.interpret_similarity(
                similarity, context=context)
            dist_interp = self.calculator.interpret_distance(
                distance, context=context)

            output = f"""
{'='*75}
ANALYSE COMPLÈTE - SIMILARITÉ DE JACCARD
{'='*75}

📝 PHRASES ANALYSÉES:
──────────────────────────────────────────────────────────────────────────
Phrase 1: "{result['sentence1']}"
Phrase 2: "{result['sentence2']}"

⚙️  CONFIGURATION ACTIVE:
──────────────────────────────────────────────────────────────────────────
• Sensibilité à la casse: {'OUI' if self.case_sensitive.get() else 'NON'}
• Suppression ponctuation: {'OUI' if self.remove_punctuation.get() else 'NON'}
• Stop-words retirés: {'OUI' if self.remove_stopwords.get() else 'NON'}
• Stemming appliqué: {'OUI' if self.use_stemming.get() else 'NON'}
• Contexte d'analyse: {context.upper()}

🔤 ANALYSE DES ENSEMBLES DE MOTS:
──────────────────────────────────────────────────────────────────────────
Ensemble 1 ({len(result['words_set1'])} mots):
  {sorted(result['words_set1'])}

Ensemble 2 ({len(result['words_set2'])} mots):
  {sorted(result['words_set2'])}

∩ INTERSECTION ({result['intersection_size']} mots communs):
  {sorted(result['intersection'])}

∪ UNION ({result['union_size']} mots uniques total):
  {sorted(result['union'])}

📊 MÉTRIQUES DE SIMILARITÉ:
──────────────────────────────────────────────────────────────────────────
Score de Similarité: {similarity:.4f} ({similarity*100:.2f}%)
Catégorie: {sim_interp['emoji']} {sim_interp['category']}

💡 Interprétation Générale:
{sim_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{sim_interp['contextual_interpretation']}

📖 Explication Technique:
{sim_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in sim_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
📏 MÉTRIQUES DE DISTANCE:
──────────────────────────────────────────────────────────────────────────
Score de Distance: {distance:.4f} ({distance*100:.2f}%)
Catégorie: {dist_interp['emoji']} {dist_interp['category']}

💡 Interprétation Générale:
{dist_interp['general_interpretation']}

🎯 Interprétation Contextuelle ({context}):
{dist_interp['contextual_interpretation']}

📖 Explication Technique:
{dist_interp['technical_explanation']}

📌 Recommandations:
"""
            for rec in dist_interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"""
✓ VÉRIFICATION MATHÉMATIQUE:
──────────────────────────────────────────────────────────────────────────
Similarité ({similarity:.4f}) + Distance ({distance:.4f}) = {similarity + distance:.4f}
Formule: J(A,B) = |A ∩ B| / |A ∪ B| = {result['intersection_size']}/{result['union_size']} = {similarity:.4f}

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status(
                f"Analyse terminée | Sim: {similarity:.4f} | Dist: {distance:.4f} | Contexte: {context}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def show_technical_details(self):
        """Affiche les détails techniques COMPLETS."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention", "Veuillez saisir les deux phrases.")
            return

        try:
            # Prétraitement détaillé
            set1 = self.calculator.preprocess_sentence(phrase1)
            set2 = self.calculator.preprocess_sentence(phrase2)

            intersection = set1.intersection(set2)
            union = set1.union(set2)
            diff1 = set1.difference(set2)
            diff2 = set2.difference(set1)

            similarity = len(intersection) / \
                len(union) if len(union) > 0 else 0.0

            output = f"""
{'='*75}
DÉTAILS TECHNIQUES COMPLETS
{'='*75}

📋 PRÉTRAITEMENT:
──────────────────────────────────────────────────────────────────────────
Phrase originale 1: "{phrase1}"
Après prétraitement: {sorted(set1)}

Phrase originale 2: "{phrase2}"
Après prétraitement: {sorted(set2)}

🔢 OPÉRATIONS SUR ENSEMBLES:
──────────────────────────────────────────────────────────────────────────
|A| = {len(set1)} mots
|B| = {len(set2)} mots

|A ∩ B| = {len(intersection)} mots
Intersection: {sorted(intersection)}

|A ∪ B| = {len(union)} mots
Union: {sorted(union)}

|A - B| = {len(diff1)} mots (uniquement dans A)
Différence A-B: {sorted(diff1)}

|B - A| = {len(diff2)} mots (uniquement dans B)
Différence B-A: {sorted(diff2)}

📐 CALCULS MATHÉMATIQUES:
──────────────────────────────────────────────────────────────────────────
Similarité de Jaccard:
J(A,B) = |A ∩ B| / |A ∪ B|
J(A,B) = {len(intersection)} / {len(union)}
J(A,B) = {similarity:.6f}

Distance de Jaccard:
d(A,B) = 1 - J(A,B)
d(A,B) = 1 - {similarity:.6f}
d(A,B) = {1-similarity:.6f}

📊 STATISTIQUES:
──────────────────────────────────────────────────────────────────────────
Taux de chevauchement: {(len(intersection)/max(len(set1), len(set2))*100) if max(len(set1), len(set2)) > 0 else 0:.2f}%
Mots communs/Phrase 1: {(len(intersection)/len(set1)*100) if len(set1) > 0 else 0:.2f}%
Mots communs/Phrase 2: {(len(intersection)/len(set2)*100) if len(set2) > 0 else 0:.2f}%

{'='*75}
"""

            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status("Détails techniques affichés")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def clear_simple_comparison(self):
        """Efface les champs."""
        self.phrase1_text.delete("1.0", tk.END)
        self.phrase2_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='normal')
        self.simple_result_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='disabled')
        self.update_status("Champs effacés")

    def create_multiple_comparison_tab(self):
        """Onglet de comparaison multiple."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Multiple  ")

        tab.rowconfigure(2, weight=1)
        tab.rowconfigure(4, weight=1)
        tab.columnconfigure(0, weight=1)

        input_frame = tk.LabelFrame(
            tab, text="📝 Ajouter des phrases",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        input_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        input_frame.columnconfigure(0, weight=1)

        tk.Label(input_frame, text="Nouvelle phrase:").grid(
            row=0, column=0, sticky='w', pady=(0, 5))

        phrase_entry_frame = tk.Frame(input_frame)
        phrase_entry_frame.grid(row=1, column=0, sticky='ew')
        phrase_entry_frame.columnconfigure(0, weight=1)

        self.multi_phrase_entry = tk.Entry(
            phrase_entry_frame, font=('Arial', 10))
        self.multi_phrase_entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.multi_phrase_entry.bind('<Return>', lambda e: self.add_phrase())

        tk.Button(
            phrase_entry_frame, text="➕ Ajouter", command=self.add_phrase,
            bg=self.colors['success'], fg='white', font=('Arial', 10, 'bold')
        ).grid(row=0, column=1)

        list_frame = tk.LabelFrame(
            tab, text="📋 Phrases à comparer",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        list_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.phrases_listbox = tk.Listbox(
            list_frame, font=('Arial', 10),
            yscrollcommand=scrollbar.set, selectmode=tk.SINGLE
        )
        self.phrases_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.phrases_listbox.yview)

        button_frame = tk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10, sticky='ew', padx=10)
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        tk.Button(
            button_frame, text="🔍 Comparer Toutes",
            command=self.compare_multiple_phrases,
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="➖ Supprimer",
            command=self.remove_phrase,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=1, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="🗑️ Tout Effacer",
            command=self.clear_all_phrases,
            bg=self.colors['warning'], fg='white',
            font=('Arial', 10, 'bold'), pady=8
        ).grid(row=0, column=2, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=4, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.multi_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.multi_result_text.pack(fill='both', expand=True)

    def add_phrase(self):
        """Ajoute une phrase."""
        phrase = self.multi_phrase_entry.get().strip()
        if not phrase:
            messagebox.showwarning("Attention", "Veuillez saisir une phrase.")
            return
        if phrase in self.phrases_list:
            messagebox.showinfo("Information", "Cette phrase existe déjà.")
            return
        self.phrases_list.append(phrase)
        self.phrases_listbox.insert(
            tk.END, f"{len(self.phrases_list)}. {phrase}")
        self.multi_phrase_entry.delete(0, tk.END)
        self.update_status(
            f"Phrase ajoutée ({len(self.phrases_list)} phrases)")

    def remove_phrase(self):
        """Supprime la phrase sélectionnée."""
        selection = self.phrases_listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "Attention", "Veuillez sélectionner une phrase.")
            return
        index = selection[0]
        del self.phrases_list[index]
        self.phrases_listbox.delete(0, tk.END)
        for i, phrase in enumerate(self.phrases_list, 1):
            self.phrases_listbox.insert(tk.END, f"{i}. {phrase}")
        self.update_status(
            f"Phrase supprimée ({len(self.phrases_list)} restantes)")

    def clear_all_phrases(self):
        """Efface toutes les phrases."""
        if self.phrases_list:
            if messagebox.askyesno("Confirmation", "Effacer toutes les phrases?"):
                self.phrases_list.clear()
                self.phrases_listbox.delete(0, tk.END)
                self.multi_result_text.config(state='normal')
                self.multi_result_text.delete("1.0", tk.END)
                self.multi_result_text.config(state='disabled')
                self.update_status("Toutes les phrases effacées")

    def compare_multiple_phrases(self):
        """Compare toutes les phrases."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = self.calculator.compare_multiple_sentences(
                self.phrases_list)
            results.sort(key=lambda x: x[2], reverse=True)

            output = f"""
{'='*70}
COMPARAISON MULTIPLE DE PHRASES
{'='*70}

Nombre de phrases: {len(self.phrases_list)}
Nombre de comparaisons: {len(results)}

{'─'*70}
TOP 10 PAIRES LES PLUS SIMILAIRES:
{'─'*70}
"""
            for i, (idx1, idx2, sim) in enumerate(results[:10], 1):
                output += f"\n{i}. Similarité: {sim:.4f}\n"
                output += f"   Phrase {idx1+1}: {self.phrases_list[idx1][:60]}...\n"
                output += f"   Phrase {idx2+1}: {self.phrases_list[idx2][:60]}...\n"

            idx1, idx2, max_sim = self.calculator.get_most_similar_pair(
                self.phrases_list)
            output += f"\n{'─'*70}\n🏆 PAIRE LA PLUS SIMILAIRE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_sim:.4f}\n"

            idx1, idx2, max_dist = self.calculator.get_most_different_pair(
                self.phrases_list)
            output += f"\n📏 PAIRE LA PLUS DIFFÉRENTE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_dist:.4f} (distance)\n"
            output += f"{'='*70}\n"

            self.multi_result_text.config(state='normal')
            self.multi_result_text.delete("1.0", tk.END)
            self.multi_result_text.insert("1.0", output)
            self.multi_result_text.config(state='disabled')

            self.update_status(f"Comparaison: {len(results)} paires analysées")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_matrix_tab(self):
        """Onglet matrices."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Matrices  ")

        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        info_label = tk.Label(
            tab,
            text="Matrices de similarité et distance pour les phrases de 'Comparaison Multiple'.",
            font=('Arial', 10), wraplength=900
        )
        info_label.grid(row=0, column=0, pady=10, padx=20, sticky='w')

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="📊 Matrice Similarité",
            command=lambda: self.generate_matrix('similarity'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Matrice Distance",
            command=lambda: self.generate_matrix('distance'),
            bg=self.colors['primary'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=10
        ).grid(row=0, column=1, padx=5, sticky='ew')

        matrix_frame = tk.LabelFrame(
            tab, text="🔢 Matrice",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        matrix_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        matrix_frame.rowconfigure(0, weight=1)
        matrix_frame.columnconfigure(0, weight=1)

        self.matrix_text = scrolledtext.ScrolledText(
            matrix_frame, font=('Courier', 9), wrap=tk.NONE, state='disabled')
        self.matrix_text.grid(row=0, column=0, sticky='nsew')

        xscrollbar = tk.Scrollbar(matrix_frame, orient='horizontal')
        xscrollbar.grid(row=1, column=0, sticky='ew')
        self.matrix_text.config(xscrollcommand=xscrollbar.set)
        xscrollbar.config(command=self.matrix_text.xview)

    def generate_matrix(self, matrix_type='similarity'):
        """Génère la matrice."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            if matrix_type == 'similarity':
                matrix = self.calculator.get_similarity_matrix(
                    self.phrases_list)
                title = "MATRICE DE SIMILARITÉ"
                legend = "Valeurs élevées = très similaires"
            else:
                matrix = self.calculator.get_distance_matrix(self.phrases_list)
                title = "MATRICE DE DISTANCE"
                legend = "Valeurs élevées = très différents"

            output = f"""
{'='*70}
{title}
{'='*70}

Phrases analysées:
"""
            for i, phrase in enumerate(self.phrases_list):
                output += f"  {i}: {phrase[:60]}...\n"

            output += f"\n{'─'*70}\nMatrice:\n\n     "
            for i in range(len(self.phrases_list)):
                output += f"{i:8}"
            output += "\n"

            for i, row in enumerate(matrix):
                output += f"{i:3}: "
                for value in row:
                    output += f"{value:8.4f}"
                output += "\n"

            output += f"\n{'='*70}\nLégende:\n"
            output += f"  • Diagonale = {'1.00' if matrix_type == 'similarity' else '0.00'}\n"
            output += f"  • {legend}\n"
            output += f"{'='*70}\n"

            self.matrix_text.config(state='normal')
            self.matrix_text.delete("1.0", tk.END)
            self.matrix_text.insert("1.0", output)
            self.matrix_text.config(state='disabled')

            self.update_status(f"Matrice {matrix_type} générée")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_extreme_pairs_tab(self):
        """NOUVEAU: Onglet paires extrêmes."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Paires Extrêmes  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="ℹ️ Information",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Recherche automatique des paires les plus similaires et les plus différentes.",
            font=('Arial', 10), wraplength=900
        ).pack()

        button_frame = tk.Frame(tab)
        button_frame.grid(row=1, column=0, pady=10, sticky='ew', padx=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame, text="🏆 Paire la Plus Similaire",
            command=self.find_most_similar,
            bg=self.colors['success'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=0, padx=5, sticky='ew')

        tk.Button(
            button_frame, text="📏 Paire la Plus Différente",
            command=self.find_most_different,
            bg=self.colors['danger'], fg='white',
            font=('Arial', 11, 'bold'), padx=20, pady=15
        ).grid(row=0, column=1, padx=5, sticky='ew')

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.extreme_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.extreme_result_text.pack(fill='both', expand=True)

    def find_most_similar(self):
        """Trouve la paire la plus similaire."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, similarity = self.calculator.get_most_similar_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_similarity(similarity)

            output = f"""
{'='*70}
🏆 PAIRE LA PLUS SIMILAIRE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📊 SCORE DE SIMILARITÉ: {similarity:.4f} ({similarity*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus similaire: {similarity:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def find_most_different(self):
        """Trouve la paire la plus différente."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            idx1, idx2, distance = self.calculator.get_most_different_pair(
                self.phrases_list)

            result = self.calculator.calculate_distance_detailed(
                self.phrases_list[idx1], self.phrases_list[idx2])

            interp = self.calculator.interpret_distance(distance)

            output = f"""
{'='*70}
📏 PAIRE LA PLUS DIFFÉRENTE
{'='*70}

Phrase {idx1+1}: "{self.phrases_list[idx1]}"
Phrase {idx2+1}: "{self.phrases_list[idx2]}"

📏 SCORE DE DISTANCE: {distance:.4f} ({distance*100:.2f}%)
{interp['emoji']} Catégorie: {interp['category']}

🔤 ANALYSE:
Mots communs ({result['intersection_size']}): {sorted(result['intersection'])}
Mots total ({result['union_size']}): {sorted(result['union'])}

💡 INTERPRÉTATION:
{interp['general_interpretation']}

📌 RECOMMANDATIONS:
"""
            for rec in interp['recommendations']:
                output += f"  • {rec}\n"

            output += f"\n{'='*70}\n"

            self.extreme_result_text.config(state='normal')
            self.extreme_result_text.delete("1.0", tk.END)
            self.extreme_result_text.insert("1.0", output)
            self.extreme_result_text.config(state='disabled')

            self.update_status(f"Paire la plus différente: {distance:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_demo_tests_tab(self):
        """NOUVEAU: Onglet tests automatiques."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Tests Auto  ")

        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        info_frame = tk.LabelFrame(
            tab, text="🧪 Tests Automatiques",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Lance des tests de démonstration comme jaccard_similarity.py",
            font=('Arial', 10)
        ).pack()

        tk.Button(
            tab, text="▶️ Lancer Tests de Démonstration",
            command=self.run_demo_tests,
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).grid(row=1, column=0, pady=20)

        result_frame = tk.LabelFrame(
            tab, text="📊 Résultats des Tests",
            font=('Arial', 10, 'bold'), padx=10, pady=10
        )
        result_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.demo_result_text = scrolledtext.ScrolledText(
            result_frame, font=('Courier', 9), wrap=tk.WORD, state='disabled')
        self.demo_result_text.pack(fill='both', expand=True)

    def run_demo_tests(self):
        """Lance les tests de démonstration."""
        examples = [
            ("Le chat mange des croquettes", "Le chien mange des croquettes"),
            ("Python est un langage de programmation",
             "Java est un langage de programmation"),
            ("Machine learning supervisé", "Apprentissage automatique supervisé"),
            ("Bonjour tout le monde", "Salut tout le monde"),
            ("Aucun mot en commun", "Différentes phrases complètement")
        ]

        output = f"""
{'='*70}
TESTS DE DÉMONSTRATION AUTOMATIQUES
{'='*70}

Configuration active:
  - Sensibilité à la casse: {'Activée' if self.case_sensitive.get() else 'Désactivée'}
  - Stop-words: {'Activés' if self.remove_stopwords.get() else 'Désactivés'}
  - Stemming: {'Activé' if self.use_stemming.get() else 'Désactivé'}

{'─'*70}
TESTS:
{'─'*70}
"""

        for i, (s1, s2) in enumerate(examples, 1):
            similarity = self.calculator.calculate_similarity(s1, s2)
            distance = self.calculator.calculate_distance(s1, s2)

            sim_interp = self.calculator.interpret_similarity(similarity)
            dist_interp = self.calculator.interpret_distance(distance)

            output += f"""
Test {i}:
  Phrase 1: '{s1}'
  Phrase 2: '{s2}'

  📊 SIMILARITÉ: {similarity:.4f}
     Catégorie: {sim_interp['emoji']} {sim_interp['category']}
     {sim_interp['general_interpretation'][:100]}...

  📏 DISTANCE: {distance:.4f}
     Catégorie: {dist_interp['emoji']} {dist_interp['category']}

  ✓ Vérification: {similarity:.4f} + {distance:.4f} = {similarity + distance:.4f}
{'─'*70}
"""

        output += f"\n{'='*70}\nTOUS LES TESTS TERMINÉS\n{'='*70}\n"

        self.demo_result_text.config(state='normal')
        self.demo_result_text.delete("1.0", tk.END)
        self.demo_result_text.insert("1.0", output)
        self.demo_result_text.config(state='disabled')

        self.update_status(f"{len(examples)} tests de démonstration terminés")

    def create_export_tab(self):
        """Onglet export."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Export  ")

        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame, text="Export des Résultats",
            font=('Arial', 16, 'bold')
        ).pack(pady=20)

        tk.Label(
            main_frame,
            text="Exportez les résultats au format CSV ou JSON",
            font=('Arial', 11), wraplength=600
        ).pack(pady=10)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=30)

        tk.Button(
            button_frame, text="💾 Exporter CSV",
            command=lambda: self.export_results('csv'),
            bg=self.colors['success'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        tk.Button(
            button_frame, text="💾 Exporter JSON",
            command=lambda: self.export_results('json'),
            bg=self.colors['secondary'], fg='white',
            font=('Arial', 12, 'bold'), padx=30, pady=15
        ).pack(side='left', padx=10)

        self.export_status = tk.Label(main_frame, text="", font=('Arial', 10))
        self.export_status.pack(pady=20)

    def export_results(self, format_type):
        """Exporte les résultats."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning("Attention", "Au moins 2 phrases requises.")
            return

        try:
            results = []
            for i in range(len(self.phrases_list)):
                for j in range(i + 1, len(self.phrases_list)):
                    detailed = self.calculator.calculate_distance_detailed(
                        self.phrases_list[i], self.phrases_list[j])
                    results.append(detailed)

            if format_type == 'csv':
                filename = self.calculator.export_results_to_csv(results)
            else:
                filename = self.calculator.export_results_to_json(results)

            if filename:
                self.export_status.config(
                    text=f"✓ Export réussi: {filename}",
                    fg=self.colors['success']
                )
                messagebox.showinfo("Succès", f"Fichier créé:\n{filename}")
            else:
                self.export_status.config(
                    text="❌ Échec de l'export",
                    fg=self.colors['danger']
                )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur:\n{str(e)}")

    def create_about_tab(self):
        """Onglet À propos."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ℹ️ À Propos  ")

        main_frame = tk.Frame(tab, bg='white')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        tk.Label(
            main_frame,
            text="Calculateur de Similarité de Jaccard",
            font=('Arial', 16, 'bold'), bg='white', fg=self.colors['primary']
        ).pack(pady=10)

        description = """
✨ Version 2.2 - Interface Graphique Complète

Cette interface implémente TOUTES les fonctionnalités de jaccard_similarity.py:

• Calcul de similarité ET distance de Jaccard
• Stemming français amélioré
• Support des stop-words français (60+)
• Interprétation contextuelle (5 contextes)
• Export CSV et JSON
• Tests automatiques de démonstration
• Recherche de paires extrêmes
• Matrices complètes
• Analyse technique détaillée
• Interface flexible et responsive

📐 Formules:
Similarité(A,B) = |A ∩ B| / |A ∪ B|
Distance(A,B) = 1 - Similarité(A,B)
        """

        tk.Label(
            main_frame, text=description, font=('Arial', 10),
            bg='white', justify='left'
        ).pack(pady=20)

        team_frame = tk.LabelFrame(
            main_frame, text="👥 Équipe",
            font=('Arial', 11, 'bold'), bg='white', padx=20, pady=15
        )
        team_frame.pack(fill='x', pady=10)

        for member in ["OUEDRAOGO Lassina", "OUEDRAOGO Rasmane", "POUBERE Abdourazakou"]:
            tk.Label(
                team_frame, text=f"• {member}",
                font=('Arial', 10), bg='white'
            ).pack(anchor='w', pady=2)

        tk.Label(
            main_frame,
            text="📚 Machine Learning non Supervisé\n🎓 Octobre 2025 - Version Complète v2.2",
            font=('Arial', 10), bg='white', fg=self.colors['dark']
        ).pack(pady=20)

    def create_status_bar(self):
        """Crée la barre de statut."""
        self.status_bar = tk.Label(
            self.root,
            text="Prêt | Interface Complète avec TOUTES les fonctionnalités",
            bd=1, relief=tk.SUNKEN, anchor='w', font=('Arial', 9)
        )
        self.status_bar.grid(row=3, column=0, sticky='ew')

    def update_status(self, message):
        """Met à jour le statut."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main():
    """Point d'entrée."""
    root = tk.Tk()
    app = JaccardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
