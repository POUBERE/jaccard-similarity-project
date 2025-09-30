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
