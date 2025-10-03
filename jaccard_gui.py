#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique pour le calculateur de similarité de Jaccard
Projet de Machine Learning non Supervisé

Auteurs: OUEDRAOGO Lassina, OUEDRAOGO Rasmane, POUBERE Abdourazakou
Date: Octobre 2025

Usage: python jaccard_gui.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List
import sys
import os

# Import du module principal
from jaccard_similarity import JaccardSimilarity


class JaccardGUI:
    """Interface graphique pour le calculateur de similarité de Jaccard."""

    def __init__(self, root):
        """Initialise l'interface graphique."""
        self.root = root
        self.root.title("Calculateur de Similarité de Jaccard")
        self.root.geometry("900x700")

        # Configuration du style
        style = ttk.Style()
        style.theme_use('clam')

        # Couleurs personnalisées
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'light': '#ECF0F1',
            'dark': '#34495E'
        }

        # Initialisation du calculateur
        self.calculator = JaccardSimilarity()

        # Variables pour les options
        self.case_sensitive = tk.BooleanVar(value=False)
        self.remove_punctuation = tk.BooleanVar(value=True)

        # Liste pour stocker les phrases multiples
        self.phrases_list = []

        # Création de l'interface
        self.create_widgets()

    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        # En-tête
        self.create_header()

        # Options de configuration
        self.create_options_frame()

        # Notebook avec onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Onglet 1: Comparaison simple
        self.create_simple_comparison_tab()

        # Onglet 2: Comparaison multiple
        self.create_multiple_comparison_tab()

        # Onglet 3: Matrice de similarité
        self.create_matrix_tab()

        # Onglet 4: À propos
        self.create_about_tab()

        # Barre de statut
        self.create_status_bar()

    def create_header(self):
        """Crée l'en-tête de l'application."""
        header_frame = tk.Frame(
            self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x', side='top')

        title_label = tk.Label(
            header_frame,
            text="📊 Calculateur de Similarité de Jaccard",
            font=('Arial', 20, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=20)

        subtitle_label = tk.Label(
            header_frame,
            text="Machine Learning non Supervisé",
            font=('Arial', 10),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        subtitle_label.pack()

    def create_options_frame(self):
        """Crée le cadre des options de configuration."""
        options_frame = tk.LabelFrame(
            self.root,
            text="⚙️ Options de Configuration",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        options_frame.pack(fill='x', padx=10, pady=5)

        # Checkboxes pour les options
        case_check = tk.Checkbutton(
            options_frame,
            text="Sensible à la casse",
            variable=self.case_sensitive,
            command=self.update_calculator
        )
        case_check.pack(side='left', padx=20)

        punct_check = tk.Checkbutton(
            options_frame,
            text="Supprimer la ponctuation",
            variable=self.remove_punctuation,
            command=self.update_calculator
        )
        punct_check.pack(side='left', padx=20)

    def update_calculator(self):
        """Met à jour le calculateur avec les nouvelles options."""
        self.calculator = JaccardSimilarity(
            case_sensitive=self.case_sensitive.get(),
            remove_punctuation=self.remove_punctuation.get()
        )
        self.update_status("Options mises à jour")

    def create_simple_comparison_tab(self):
        """Crée l'onglet de comparaison simple."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Simple  ")

        # Frame principal avec padding
        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Phrase 1
        tk.Label(
            main_frame,
            text="Phrase 1:",
            font=('Arial', 11, 'bold')
        ).grid(row=0, column=0, sticky='w', pady=5)

        self.phrase1_text = scrolledtext.ScrolledText(
            main_frame,
            height=4,
            width=70,
            font=('Arial', 10),
            wrap=tk.WORD
        )
        self.phrase1_text.grid(row=1, column=0, pady=5)

        # Phrase 2
        tk.Label(
            main_frame,
            text="Phrase 2:",
            font=('Arial', 11, 'bold')
        ).grid(row=2, column=0, sticky='w', pady=5)

        self.phrase2_text = scrolledtext.ScrolledText(
            main_frame,
            height=4,
            width=70,
            font=('Arial', 10),
            wrap=tk.WORD
        )
        self.phrase2_text.grid(row=3, column=0, pady=5)

        # Boutons d'action
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=4, column=0, pady=15)

        calculate_btn = tk.Button(
            button_frame,
            text="🔍 Calculer la Similarité",
            command=self.calculate_simple_similarity,
            bg=self.colors['secondary'],
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        calculate_btn.pack(side='left', padx=5)

        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Effacer",
            command=self.clear_simple_comparison,
            bg=self.colors['warning'],
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        clear_btn.pack(side='left', padx=5)

        # Zone de résultats
        result_frame = tk.LabelFrame(
            main_frame,
            text="📊 Résultats",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=10
        )
        result_frame.grid(row=5, column=0, pady=10, sticky='ew')

        self.simple_result_text = scrolledtext.ScrolledText(
            result_frame,
            height=12,
            width=70,
            font=('Courier', 10),
            wrap=tk.WORD,
            state='disabled'
        )
        self.simple_result_text.pack()

    def create_multiple_comparison_tab(self):
        """Crée l'onglet de comparaison multiple."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Comparaison Multiple  ")

        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Zone de saisie
        input_frame = tk.LabelFrame(
            main_frame,
            text="📝 Ajouter des phrases",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=10
        )
        input_frame.pack(fill='x', pady=10)

        tk.Label(input_frame, text="Nouvelle phrase:").pack(anchor='w')

        phrase_entry_frame = tk.Frame(input_frame)
        phrase_entry_frame.pack(fill='x', pady=5)

        self.multi_phrase_entry = tk.Entry(
            phrase_entry_frame,
            font=('Arial', 10),
            width=60
        )
        self.multi_phrase_entry.pack(side='left', padx=5)
        self.multi_phrase_entry.bind('<Return>', lambda e: self.add_phrase())

        add_btn = tk.Button(
            phrase_entry_frame,
            text="➕ Ajouter",
            command=self.add_phrase,
            bg=self.colors['success'],
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2'
        )
        add_btn.pack(side='left')

        # Liste des phrases
        list_frame = tk.LabelFrame(
            main_frame,
            text="📋 Phrases à comparer",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=10
        )
        list_frame.pack(fill='both', expand=True, pady=10)

        # Listbox avec scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.phrases_listbox = tk.Listbox(
            list_frame,
            font=('Arial', 10),
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE
        )
        self.phrases_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.phrases_listbox.yview)

        # Boutons d'action
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        compare_btn = tk.Button(
            button_frame,
            text="🔍 Comparer Toutes",
            command=self.compare_multiple_phrases,
            bg=self.colors['secondary'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        compare_btn.pack(side='left', padx=5)

        remove_btn = tk.Button(
            button_frame,
            text="➖ Supprimer",
            command=self.remove_phrase,
            bg=self.colors['danger'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        remove_btn.pack(side='left', padx=5)

        clear_all_btn = tk.Button(
            button_frame,
            text="🗑️ Tout Effacer",
            command=self.clear_all_phrases,
            bg=self.colors['warning'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8,
            cursor='hand2'
        )
        clear_all_btn.pack(side='left', padx=5)

        # Zone de résultats
        result_frame = tk.LabelFrame(
            main_frame,
            text="📊 Résultats",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=10
        )
        result_frame.pack(fill='both', expand=True, pady=10)

        self.multi_result_text = scrolledtext.ScrolledText(
            result_frame,
            height=10,
            font=('Courier', 9),
            wrap=tk.WORD,
            state='disabled'
        )
        self.multi_result_text.pack(fill='both', expand=True)

    def create_matrix_tab(self):
        """Crée l'onglet de matrice de similarité."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Matrice de Similarité  ")

        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        info_label = tk.Label(
            main_frame,
            text="Cette vue affiche la matrice de similarité pour les phrases ajoutées dans l'onglet 'Comparaison Multiple'.",
            font=('Arial', 10),
            wraplength=700,
            justify='left'
        )
        info_label.pack(pady=10)

        # Bouton pour générer la matrice
        generate_btn = tk.Button(
            main_frame,
            text="📊 Générer la Matrice",
            command=self.generate_matrix,
            bg=self.colors['secondary'],
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        generate_btn.pack(pady=10)

        # Zone d'affichage de la matrice
        matrix_frame = tk.LabelFrame(
            main_frame,
            text="🔢 Matrice de Similarité",
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=10
        )
        matrix_frame.pack(fill='both', expand=True, pady=10)

        self.matrix_text = scrolledtext.ScrolledText(
            matrix_frame,
            font=('Courier', 9),
            wrap=tk.NONE,
            state='disabled'
        )
        self.matrix_text.pack(fill='both', expand=True)

        # Scrollbar horizontale
        xscrollbar = tk.Scrollbar(matrix_frame, orient='horizontal')
        xscrollbar.pack(side='bottom', fill='x')
        self.matrix_text.config(xscrollcommand=xscrollbar.set)
        xscrollbar.config(command=self.matrix_text.xview)

    def create_about_tab(self):
        """Crée l'onglet À propos."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ℹ️ À Propos  ")

        main_frame = tk.Frame(tab, bg='white')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # Titre
        title_label = tk.Label(
            main_frame,
            text="Calculateur de Similarité de Jaccard",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg=self.colors['primary']
        )
        title_label.pack(pady=10)

        # Description
        description = """
        Ce projet implémente un calculateur de similarité de Jaccard
        pour comparer des phrases en utilisant l'approche "sac de mots".
        
        📐 Formule de Jaccard:
        Similarité(A,B) = |A ∩ B| / |A ∪ B|
        
        Où A et B sont les ensembles de mots des deux phrases.
        """

        desc_label = tk.Label(
            main_frame,
            text=description,
            font=('Arial', 11),
            bg='white',
            justify='left'
        )
        desc_label.pack(pady=20)

        # Informations sur l'équipe
        team_frame = tk.LabelFrame(
            main_frame,
            text="👥 Équipe",
            font=('Arial', 11, 'bold'),
            bg='white',
            padx=20,
            pady=15
        )
        team_frame.pack(fill='x', pady=10)

        team_members = [
            "OUEDRAOGO Lassina",
            "OUEDRAOGO Rasmane",
            "POUBERE Abdourazakou"
        ]

        for member in team_members:
            member_label = tk.Label(
                team_frame,
                text=f"• {member}",
                font=('Arial', 10),
                bg='white'
            )
            member_label.pack(anchor='w', pady=2)

        # Informations sur le cours
        course_label = tk.Label(
            main_frame,
            text="📚 Machine Learning non Supervisé\n🎓 Septembre 2025",
            font=('Arial', 10),
            bg='white',
            fg=self.colors['dark']
        )
        course_label.pack(pady=20)

        # Liens
        link_frame = tk.Frame(main_frame, bg='white')
        link_frame.pack(pady=10)

        github_label = tk.Label(
            link_frame,
            text="🔗 GitHub: github.com/POUBERE/jaccard-similarity-project",
            font=('Arial', 9),
            bg='white',
            fg=self.colors['secondary'],
            cursor='hand2'
        )
        github_label.pack()

    def create_status_bar(self):
        """Crée la barre de statut."""
        self.status_bar = tk.Label(
            self.root,
            text="Prêt",
            bd=1,
            relief=tk.SUNKEN,
            anchor='w',
            font=('Arial', 9)
        )
        self.status_bar.pack(side='bottom', fill='x')

    def update_status(self, message):
        """Met à jour le message de la barre de statut."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def calculate_simple_similarity(self):
        """Calcule la similarité entre deux phrases."""
        phrase1 = self.phrase1_text.get("1.0", tk.END).strip()
        phrase2 = self.phrase2_text.get("1.0", tk.END).strip()

        if not phrase1 or not phrase2:
            messagebox.showwarning(
                "Attention",
                "Veuillez saisir les deux phrases à comparer."
            )
            return

        try:
            # Calcul détaillé
            result = self.calculator.calculate_similarity_detailed(
                phrase1, phrase2)

            # Formatage du résultat
            output = f"""
{'='*60}
RÉSULTAT DE LA COMPARAISON
{'='*60}

Phrase 1: "{result['sentence1']}"
Phrase 2: "{result['sentence2']}"

{'─'*60}
ANALYSE DES MOTS
{'─'*60}

Mots Phrase 1: {sorted(result['words_set1'])}
Nombre de mots: {len(result['words_set1'])}

Mots Phrase 2: {sorted(result['words_set2'])}
Nombre de mots: {len(result['words_set2'])}

{'─'*60}
CALCUL DE LA SIMILARITÉ
{'─'*60}

Intersection (mots communs): {sorted(result['intersection'])}
Taille de l'intersection: {result['intersection_size']}

Union (tous les mots): {sorted(result['union'])}
Taille de l'union: {result['union_size']}

{'─'*60}
SIMILARITÉ DE JACCARD: {result['jaccard_similarity']:.4f}
{'='*60}

INTERPRÉTATION:
"""

            # Interprétation
            sim = result['jaccard_similarity']
            if sim == 1.0:
                interpretation = "✅ Phrases identiques"
            elif sim >= 0.8:
                interpretation = "✅ Très similaires"
            elif sim >= 0.5:
                interpretation = "⚠️  Moyennement similaires"
            elif sim > 0:
                interpretation = "❌ Peu similaires"
            else:
                interpretation = "❌ Aucune similarité"

            output += interpretation

            # Affichage
            self.simple_result_text.config(state='normal')
            self.simple_result_text.delete("1.0", tk.END)
            self.simple_result_text.insert("1.0", output)
            self.simple_result_text.config(state='disabled')

            self.update_status(f"Similarité calculée: {sim:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du calcul:\n{str(e)}")

    def clear_simple_comparison(self):
        """Efface les champs de comparaison simple."""
        self.phrase1_text.delete("1.0", tk.END)
        self.phrase2_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='normal')
        self.simple_result_text.delete("1.0", tk.END)
        self.simple_result_text.config(state='disabled')
        self.update_status("Champs effacés")

    def add_phrase(self):
        """Ajoute une phrase à la liste."""
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
        self.phrases_listbox.delete(index)
        del self.phrases_list[index]

        # Réinitialiser la listbox
        self.phrases_listbox.delete(0, tk.END)
        for i, phrase in enumerate(self.phrases_list, 1):
            self.phrases_listbox.insert(tk.END, f"{i}. {phrase}")

        self.update_status(
            f"Phrase supprimée ({len(self.phrases_list)} phrases restantes)")

    def clear_all_phrases(self):
        """Efface toutes les phrases."""
        if self.phrases_list:
            if messagebox.askyesno("Confirmation", "Voulez-vous vraiment effacer toutes les phrases?"):
                self.phrases_list.clear()
                self.phrases_listbox.delete(0, tk.END)
                self.multi_result_text.config(state='normal')
                self.multi_result_text.delete("1.0", tk.END)
                self.multi_result_text.config(state='disabled')
                self.update_status("Toutes les phrases effacées")

    def compare_multiple_phrases(self):
        """Compare toutes les phrases de la liste."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning(
                "Attention",
                "Veuillez ajouter au moins 2 phrases à comparer."
            )
            return

        try:
            # Comparaison de toutes les paires
            results = self.calculator.compare_multiple_sentences(
                self.phrases_list)
            results.sort(key=lambda x: x[2], reverse=True)

            # Formatage du résultat
            output = f"""
{'='*70}
COMPARAISON MULTIPLE DE PHRASES
{'='*70}

Nombre de phrases: {len(self.phrases_list)}
Nombre de comparaisons: {len(results)}

{'─'*70}
TOP 10 DES PAIRES LES PLUS SIMILAIRES:
{'─'*70}

"""

            for i, (idx1, idx2, sim) in enumerate(results[:10], 1):
                output += f"\n{i}. Similarité: {sim:.4f}\n"
                output += f"   Phrase {idx1+1}: {self.phrases_list[idx1][:60]}...\n"
                output += f"   Phrase {idx2+1}: {self.phrases_list[idx2][:60]}...\n"

            # Paire la plus similaire
            idx1, idx2, max_sim = self.calculator.get_most_similar_pair(
                self.phrases_list)

            output += f"\n{'─'*70}\n"
            output += f"PAIRE LA PLUS SIMILAIRE:\n"
            output += f"Phrases {idx1+1} et {idx2+1}: {max_sim:.4f}\n"
            output += f"{'='*70}\n"

            # Affichage
            self.multi_result_text.config(state='normal')
            self.multi_result_text.delete("1.0", tk.END)
            self.multi_result_text.insert("1.0", output)
            self.multi_result_text.config(state='disabled')

            self.update_status(
                f"Comparaison terminée: {len(results)} paires analysées")

        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Erreur lors de la comparaison:\n{str(e)}")

    def generate_matrix(self):
        """Génère et affiche la matrice de similarité."""
        if len(self.phrases_list) < 2:
            messagebox.showwarning(
                "Attention",
                "Veuillez ajouter au moins 2 phrases pour générer une matrice."
            )
            return

        try:
            matrix = self.calculator.get_similarity_matrix(self.phrases_list)

            # Formatage de la matrice
            output = f"""
{'='*70}
MATRICE DE SIMILARITÉ
{'='*70}

Phrases analysées:
"""

            for i, phrase in enumerate(self.phrases_list):
                output += f"  {i}: {phrase[:60]}...\n"

            output += f"\n{'─'*70}\n"
            output += "Matrice:\n\n"

            # En-tête
            output += "     "
            for i in range(len(self.phrases_list)):
                output += f"{i:8}"
            output += "\n"

            # Lignes de la matrice
            for i, row in enumerate(matrix):
                output += f"{i:3}: "
                for sim in row:
                    output += f"{sim:8.4f}"
                output += "\n"

            output += f"\n{'='*70}\n"
            output += "Légende:\n"
            output += "  • Diagonale = 1.00 (phrase identique à elle-même)\n"
            output += "  • Valeurs élevées = phrases très similaires\n"
            output += "  • Valeurs faibles = phrases peu similaires\n"
            output += f"{'='*70}\n"

            # Affichage
            self.matrix_text.config(state='normal')
            self.matrix_text.delete("1.0", tk.END)
            self.matrix_text.insert("1.0", output)
            self.matrix_text.config(state='disabled')

            self.update_status(
                f"Matrice générée ({len(self.phrases_list)}x{len(self.phrases_list)})")

        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Erreur lors de la génération:\n{str(e)}")


def main():
    """Point d'entrée de l'application."""
    root = tk.Tk()
    app = JaccardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
