#
# preprocess_speeches.py
# Tokenizes and lemmantizes speeches by the two presidents
# Last Modified: 8/20/2017
# Modified By: Andrew Roberts
#

import nltk
import numpy as np
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

lemmantizer = WordNetLemmatizer()

