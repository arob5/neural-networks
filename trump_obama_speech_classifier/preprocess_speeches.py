#
# preprocess_speeches.py
# Tokenizes and lemmantizes speeches by the two presidents
# Last Modified: 8/21/2017
# Modified By: Andrew Roberts
#
#
# Trump speech .txt file contains 213,801 tokens
# Obama speech .txt file contains 178,867 tokens
#

import nltk
import math
import random
import pickle
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def create_feature_label_sets(file1, file2, test_size=.1):
	lexicon = create_lexicon([file1, file2])
	features = []
	features += parse_observation(file1, lexicon, [1, 0])
	features += parse_observation(file2, lexicon, [0, 1])
	random.shuffle(features)
	
	features = np.array(features)
	n_test_examples = int(test_size * len(features))

	x_train = list(features[:, 0][: -n_test_examples])
	y_train = list(features[:, 1][: -n_test_examples])
	
	x_test = list(features[:, 0][-n_test_examples:])
	y_test = list(features[:, 1][-n_test_examples:])

	return x_train, x_test, y_train, y_test

def create_lexicon(text_files):
	lexicon = []
	for file in text_files:
		with open(file, "r") as f:
			contents = f.readlines()

			for line in contents:
				if not line.strip():
					continue	
				all_words = word_tokenize(line.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	word_counts = Counter(lexicon)

	lexicon_trimmed = []
	for token in word_counts:
		if 1000 > word_counts[token] > 50:
			lexicon_trimmed.append(token)
	
	return lexicon_trimmed

def parse_observation(sample, lexicon, classification):
	feature_set = []

	with open(sample, "r") as f:
		contents = f.readlines()
		for l in contents:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			feature_set.append([list(features), classification])
	
	return feature_set


def check_distribution(word_counts):
	""" Look at distribution of word counts; used to make cutoff decisions for word 
            counts in the create_lexicon function

	Args: word_counts (collections.Counter() object)  
	"""

	print(word_counts.most_common(100))
	word_count_values = list(word_counts.values())

	print("Word count percentiles from 10 to 100:")
	print(np.percentile(word_count_values, range(10, 100, 10)))


	

