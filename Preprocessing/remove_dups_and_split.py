import re
import num2words
from pathlib import Path
from tqdm.auto import tqdm
import time

import numpy as np
from os.path import exists

def read_file_and_strip(filename):
	lines = []
	with open(filename) as file:
		for line in file:
			lines.append(line.strip())
	return np.asarray(lines)

def remove_duplicates(pos_or_neg_string):
	NAME_OF_FILE_BEFORE_PREPROCESSING = "./" + DIRECTORY_NAME + "/train_" + pos_or_neg_string + "_full.txt"
	NAME_OF_FILE_AFTER_PREPROCESSING = "./" + DIRECTORY_NAME + "/train_" + pos_or_neg_string + "_full_no_dups.txt"

	sentences = read_file_and_strip(NAME_OF_FILE_BEFORE_PREPROCESSING)
	sorted_sentences = np.unique(sentences)
	np.savetxt(NAME_OF_FILE_AFTER_PREPROCESSING, sorted_sentences, fmt="%s")

def do_data_split():
	dataset_path = "./" + DIRECTORY_NAME + "/"
	train_neg_full_path = "train_neg_full_no_dups.txt"
	train_pos_full_path = "train_pos_full_no_dups.txt"
	    
	print("do splitting on unsplitted files")
	sentences = []
	labels = []

	def read_file(filename, label):
	    with open(filename) as file:
	        for line in file:
	            sentences.append(line.strip())
	            labels.append(label)

	read_file(dataset_path + train_neg_full_path, 0)
	read_file(dataset_path + train_pos_full_path, 1)

	sentences = np.asarray(sentences)
	labels = np.asarray(labels).astype(int)

	# do a 90%/10% train/validation split
	np.random.seed(1)
	shuffled_indices = np.random.permutation(len(sentences))
	split_index = int(0.9 * len(sentences))
	train_indices = shuffled_indices[:split_index]
	train_sentences, train_labels = sentences[train_indices], labels[train_indices]
	val_indices = shuffled_indices[split_index:]
	val_sentences, val_labels = sentences[val_indices], labels[val_indices]

	np.savetxt(dataset_path + "train_sentences.txt", train_sentences, fmt="%s")
	np.savetxt(dataset_path + "train_labels.txt", train_labels, fmt="%d")
	np.savetxt(dataset_path + "val_sentences.txt", val_sentences, fmt="%s")
	np.savetxt(dataset_path + "val_labels.txt", val_labels, fmt="%d")


directories = [
"no-stemming_no-lemmatize_no-stopwords_no-spellcorrect",
"no-stemming_no-lemmatize_with-stopwords_no-spellcorrect",
"no-stemming_no-lemmatize_with-stopwords_with-spellcorrect",
"no-stemming_with-lemmatize_with-stopwords_no-spellcorrect",
"no-stemming_with-lemmatize_with-stopwords_with-spellcorrect",
"raw",
"with-stemming_no-lemmatize_with-stopwords_no-spellcorrect",
"with-stemming_with-lemmatize_no-stopwords_with-spellcorrect",
"with-stemming_with-lemmatize_with-stopwords_no-spellcorrect",
]

for directory in directories:
	DIRECTORY_NAME = directory
	print(directory)
	remove_duplicates("neg")
	remove_duplicates("pos")
	do_data_split()