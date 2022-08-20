#!/usr/bin/env python3

from __future__ import division # avoids python2 int division

import collections
import datetime
import enum
import itertools
import math
import os
import random
import re
import stat
import statistics
import subprocess
import sys
import time

import matplotlib.pyplot as plt

from benchmarks import PSEUDOWEIGHTED_VAR_COUNTS

################################################################################
'''Defines constants.'''

FLOAT_FORMAT = '{:8.2f}'

INF = math.inf # overflow
NAN = math.nan # Counter.CACHET: math.isinf(scale) and modelCount == 0.

################################################################################
'''Defines paths.'''

EVAL_PATH = os.path.dirname(os.path.abspath(__file__))
SLURM_ARRAY_SBATCH_PATH = os.path.join(EVAL_PATH, 'SlurmArray.sbatch')
ANALYSIS_PATH = os.path.join(EVAL_PATH, 'analysis')
FIGURES_PATH = os.path.join(ANALYSIS_PATH, 'figures')

DATA_PATH = os.path.join(EVAL_PATH, 'data')
ALTOGETHER_DATA_PATH = os.path.join(DATA_PATH, 'altogether')
EXP_1_DATA_PATH = os.path.join(ALTOGETHER_DATA_PATH, 'exp1')
EXP_1_B_DATA_PATH = os.path.join(ALTOGETHER_DATA_PATH, 'exp1b')
EXP_2_DATA_PATH = os.path.join(ALTOGETHER_DATA_PATH, 'exp2')
MAVC_DATA_PATH = os.path.join(ALTOGETHER_DATA_PATH, 'mavc')

## exp1:
OUT_DIR_PATH_EXP_1_ADDMC_BAYES_MINIC2D = os.path.join(EXP_1_DATA_PATH, 'exp1-addmc-bayes_MINIC2D')
OUT_DIR_PATH_EXP_1_ADDMC_BAYES_PSEUDOWEIGHTED = os.path.join(EXP_1_DATA_PATH, 'exp1-addmc-pseudoweighted_MINIC2D')

## exp1b:
OUT_DIR_PATH_EXP_1_B_ADDMC_BAYES_MINIC2D = os.path.join(EXP_1_B_DATA_PATH, 'exp1b-addmc-bayes_MINIC2D')
OUT_DIR_PATH_EXP_1_B_ADDMC_BAYES_PSEUDOWEIGHTED = os.path.join(EXP_1_B_DATA_PATH, 'exp1b-addmc-pseudoweighted_MINIC2D')

## exp2 BAYES:
OUT_DIR_PATH_EXP_2_ADDMC_BAYES_MINIC2D = os.path.join(EXP_2_DATA_PATH, 'addmc', 'exp2-addmc-bayes_MINIC2D')
OUT_DIR_PATH_EXP_2_C2D_BAYES_DDNNF = os.path.join(EXP_2_DATA_PATH, 'c2d', 'exp2-c2d-bayes_DDNNF')
OUT_DIR_PATH_EXP_2_CACHET_BAYES_CACHET = os.path.join(EXP_2_DATA_PATH, 'cachet', 'exp2-cachet-bayes_CACHET')
OUT_DIR_PATH_EXP_2_D4_BAYES_DDNNF = os.path.join(EXP_2_DATA_PATH, 'd4', 'exp2-d4-bayes_DDNNF')
OUT_DIR_PATH_EXP_2_MINIC2D_BAYES_MINIC2D = os.path.join(EXP_2_DATA_PATH, 'minic2d', 'exp2-minic2d-bayes_MINIC2D')

## exp2 PSEUDOWEIGHTED:
OUT_DIR_PATH_EXP_2_ADDMC_PSEUDOWEIGHTED_MINIC2D = os.path.join(EXP_2_DATA_PATH, 'addmc', 'exp2-addmc-pseudoweighted_MINIC2D')
OUT_DIR_PATH_EXP_2_C2D_PSEUDOWEIGHTED_DDNNF = os.path.join(EXP_2_DATA_PATH, 'c2d', 'exp2-c2d-pseudoweighted_DDNNF')
OUT_DIR_PATH_EXP_2_CACHET_PSEUDOWEIGHTED_CACHET = os.path.join(EXP_2_DATA_PATH, 'cachet', 'exp2-cachet-pseudoweighted_CACHET')
OUT_DIR_PATH_EXP_2_D4_PSEUDOWEIGHTED_DDNNF = os.path.join(EXP_2_DATA_PATH, 'd4', 'exp2-d4-pseudoweighted_DDNNF')
OUT_DIR_PATH_EXP_2_MINIC2D_PSEUDOWEIGHTED_MINIC2D = os.path.join(EXP_2_DATA_PATH, 'minic2d', 'exp2-minic2d-pseudoweighted_MINIC2D')

ADDMC_ROOT_PATH = os.path.abspath(os.path.join(EVAL_PATH, '..'))
ADDMC_EXECUTABLE_PATH = os.path.join(ADDMC_ROOT_PATH, 'counting', 'addmc')

BENCHMARKS_PATH = os.path.join(ADDMC_ROOT_PATH, 'benchmarks')
BENCHMARKS_ALTOGETHER_PATH = os.path.join(BENCHMARKS_PATH, 'altogether')
BAYES_BENCHMARKS_PATH = os.path.join(BENCHMARKS_ALTOGETHER_PATH, 'bayes')
PSEUDOWEIGHTED_BENCHMARKS_PATH = os.path.join(BENCHMARKS_ALTOGETHER_PATH, 'pseudoweighted')
