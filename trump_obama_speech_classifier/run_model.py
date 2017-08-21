#
# run_model.py
# Trains Neural Net on president speech data
# Last Modified: 8/21/2017
# Modified By: Andrew Roberts
#

import preprocess_speeches
import numpy as np

def main():
	x_train, x_test, y_train, y_test = preprocess_speeches.create_feature_label_sets("obama_speech_transcripts.txt", "trump_speech_transcripts.txt")
	
	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

main()
