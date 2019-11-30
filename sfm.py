'''
This classes implement the algorithm described in the following paper:
Zou, Xiaorong:  "Structured Factor Model and its Applications on Market Risk Management"

the paper can be found in:   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3180413>
'''

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import minimize
#to be updated
