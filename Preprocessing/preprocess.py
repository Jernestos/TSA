import re
import num2words
from pathlib import Path
from tqdm.auto import tqdm
import time

import pkg_resources
from symspellpy import SymSpell

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
#from nltk.stem.porter import PorterStemmer #unused
from nltk.stem.snowball import SnowballStemmer #better than above used stemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

import unicodedata

import numpy as np
from os.path import exists

URL = "<url>"
USER = "<user>"
# HASHTAG = "#"
# AT = "@"
TAB = "\t"

SPELLING_CORRECTION = False
STOPWORD_REMOVAL = True
STEMMING_CORRECTION = False
LEMMATIZE_FLAG = False
DIRECTORY_NAME = "no-stemming_no-lemma_no-spell_corr_no-stopwords"


#stemmer = PorterStemmer() #for stemming, unused
stemmer = SnowballStemmer("english") #better than the above stemmer

def get_wordnet_tag(pos_tag):
    if pos_tag.startswith('V'):
        return "v"
    elif pos_tag.startswith('J'):
        return "a"
    elif pos_tag.startswith('R'):
        return "r"
    return "n"

word_net_lemmatizer = WordNetLemmatizer()

#based on paper
#https://www.researchgate.net/publication/311615347_A_Comparison_between_Preprocessing_Techniques_for_Sentiment_Analysis_in_Twitter


#not really proprocessing, but still interesting reads
#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#https://machinelearningmastery.com/clean-text-machine-learning-python/
#http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/
#https://www.aiperspectives.com/twitter-sentiment-analysis/

#21/06/22: https://en.wikipedia.org/wiki/List_of_emoticons
"""
This function was used to add spaces in the emoji characters.
def add_spaces(emoji_list): #list of emojis
     liste = []
     for emoji in some_list:
             liste.append(" ".join(emoji)) #add spaces between characters that make up emoji
     return(liste)
"""
#non-comprehensive list of emojis
#Assume that we can divide emojis between positive sentiment and negative sentiment
#E.g. emoji for skeptical is rather neutral than negative
SMILEY_EMOJI = [":‑)", ":)", ":-]", ":]", ":->", ":>""8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)", ":-))"] + [': ‑ )', ': )', ': - ]', ': ]', ': - >', ': > 8 - )', '8 )', ': - }', ': }', ': o )', ': c )', ': ^ )', '= ]', '= )', ': - ) )']
LAUGHING_EMOJI = [":-D",":D","8-D","8D","=D","=3","B^D","c:","C:","X-D","xD","X-D","XD"] + [': - D', ': D', '8 - D', '8 D', '= D', '= 3', 'B ^ D', 'c :', 'C :', 'X - D', 'x D', 'X - D', 'X D'] + ["xd", "x d", "Xd", "X d"] #added manually after reviewing training file
SAD_EMOJI = [":-(",":(",":-c",":c",":-<",":<",":-[",":[",":-||",">:[",":{",":@",":(",";(", ":/", ";/", ";\\", ":\\"] + [': - (', ': (', ': - c', ': c', ': - <', ': <', ': - [', ': [', ': - | |', '> : [', ': {', ': @', ': (', '; ('] + [': - C', ': C', ':-C', ':C'] #added manually
CRYING_EMOJI = [":'-(",":'(",":=("] + [": ' - (", ": ' (", ': = (']
HAPPINESS_EMOJI = [":'‑)",":')",":\"D"] + [": ' ‑ )", ": ' )", ': " D']
HORROR_EMOJI = ["D-':","D:<","D:","D8","D;","D=","DX"] + ["D - ' :", 'D : <', 'D :', 'D 8', 'D ;', 'D =', 'D X']
KISS_EMOJI = [":-*",":*",":x"] + [': - *', ': *', ': x']
WINK_EMOJI = [";‑)",";)","*-)","*)",";‑]",";]",";^)",";>",":-,",";D",";3"] + ['; ‑ )', '; )', '* - )', '* )', '; ‑ ]', '; ]', '; ^ )', '; >', ': - ,', '; D', '; 3']
HEART_EMOJI = ["<3"] + ["< 3"]
BROKENHEART_EMOJI = ["</3", "<\3"] + ["< / 3", "< \ 3"] 

#list of (emojilist and its' name)
#stemming already done here
EMOJI_LIST = [(SMILEY_EMOJI, "smile"), (LAUGHING_EMOJI, "laugh"), (SAD_EMOJI, "sad"), (CRYING_EMOJI, "cry"), (HAPPINESS_EMOJI, "happy"), (HORROR_EMOJI, "horror"), (KISS_EMOJI, "kiss"), (WINK_EMOJI, "wink"), (HEART_EMOJI, "love"), (BROKENHEART_EMOJI, "heartbroken")]

# Misspeling correction setup
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
		"symspellpy", "frequency_dictionary_en_82_765.txt"
)
bigram_path = pkg_resources.resource_filename(
		"symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

#next 3 methods unused
#This method removes stopwords
def stopword_removal(token_list):
	stopword_set = set(stopwords.words('english'))
	new_token_list = [token for token in token_list if not token in stopword_set]
	return new_token_list

#This method performs stemming
def stemming(token_list):
	token_list = list(map(stemmer.stem, token_list))
	new_string = " ".join(token_list) #Build string out of tokens
	return new_string

def lemmatize(token_list):
	token_list = list(map(word_net_lemmatizer.lemmatize, token_list))
	new_string = " ".join(token_list)
	return new_string
 
#helper function for stopwords removal and stemming
def stopwords_removal_n_stemmer_helper(string_to_process):
	if not STOPWORD_REMOVAL:
		return string_to_process
	token_list = word_tokenize(string_to_process) #tokenize string to remove stop words and perform stemming
	new_token_list = stopword_removal(token_list)
	if STEMMING_CORRECTION:
		return stemming(new_token_list)
	new_string = " ".join(new_token_list) #Build string out of tokens
	return new_string


def stopword_removal_stemming_lemmatize_helper(token_list):
	stopword_set = set(stopwords.words('english'))
	new_token_list = token_list
	if STOPWORD_REMOVAL:
		new_token_list = [token for token in new_token_list if not token in stopword_set] #stopword removal
	if STEMMING_CORRECTION:
		new_token_list = [stemmer.stem(token) for token in new_token_list] #stemming
	if LEMMATIZE_FLAG:
		pos_tags = pos_tag(new_token_list)
		new_token_list = [word_net_lemmatizer.lemmatize(token, pos=get_wordnet_tag(pos)) for (token, pos) in pos_tags] #lemmatize
	return " ".join(new_token_list)

#substitute emoji by its name
def emoji_substitution(string_to_process):
	new_string = string_to_process
	for (emoji_list, name_of_emoji) in EMOJI_LIST: #iterate over elements of EMOJI_LIST
		for emoji in emoji_list: #iterate over all emojis in emoji_list
			new_string = new_string.replace(emoji, name_of_emoji) #replace emoji with its' name
	return new_string
	
#preprocess evert tweet		
def filtering_strings(string_to_process):

	#map non-ascii characters to their ascii counterpart.
	new_string = unicodedata.normalize('NFKD', string_to_process).encode('ascii', 'ignore').decode()

	new_string = string_to_process.replace(URL, "") #remove <url>
	new_string = new_string.replace(USER, "") #remove <user>

	#if a character occurs >= 3 times consecutively, them limit it to 2 times
	new_string = re.sub(r"(.)\1+", r"\1\1", new_string) 
	
	new_string = emoji_substitution(new_string) #substitute emojis by their names

	#replace numbers by "number"
	new_string = re.sub(r"\d+", "number", new_string)
 
	new_string = re.sub(r"[^A-Za-z]+",' ', new_string) #substitute special characters like punctuation, !, tabs, etc. by 1 whitespace
	#new_string = new_string.replace(TAB, " ") #substitute tab for space

	#going through the training data, 3 consecutive white spaces is the maximum used in a tweet
	#already searches for 3 consecutive white spaces above, to just compress double whitespaces to 1 whitespace
	#new_string = new_string.replace("  ", " ") #remove consecutive white spaces
	new_string = re.sub(r"\s+", ' ', new_string)
 
	#combine next 2 steps as "remove special characters."
	# new_string = new_string.replace(HASHTAG, "") #remove #
	# new_string = new_string.replace(AT, "") #remove @

	#From https://stackoverflow.com/questions/40040177/search-and-replace-numbers-with-words-in-file
	#new_string = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), new_string)

	new_string = new_string.lower() #convert to lower case

	#word_tokenize(new_string)
	#new_string = stopwords_removal_n_stemmer_helper(new_string) #OLD
	new_string = stopword_removal_stemming_lemmatize_helper(word_tokenize(new_string))
 
	if(SPELLING_CORRECTION):
		suggestion = sym_spell.lookup_compound(
      new_string, max_edit_distance=2
  	)
		new_string = suggestion[0].term

	return new_string

def preprocess_file(pos_or_neg_string):
	#somewhere option to select file by specifing its name.
	NAME_OF_FILE_BEFORE_PREPROCESSING = "./raw/train_" + pos_or_neg_string + "_full.txt"
	NAME_OF_FILE_AFTER_PREPROCESSING = "./" + DIRECTORY_NAME + "/train_" + pos_or_neg_string + "_full.txt"

	sentences = []

	# Read the data 
	with open(NAME_OF_FILE_BEFORE_PREPROCESSING, 'r', encoding="utf-8") as f:
		content = f.readlines()
		#print(content[:10])
		progress_bar = tqdm(range(len(content)))
		for line in content:
			sentences.append(filtering_strings(line))
			progress_bar.update(1)

	# Write the processed data
	with open(NAME_OF_FILE_AFTER_PREPROCESSING, 'w', encoding="utf-8") as f:
		for line in sentences:
			f.write(line + "\n")

def read_file_and_strip(filename):
	lines = []
	with open(filename) as file:
		for line in file:
			lines.append(line.strip())
	return np.asarray(lines)

def do_data_split():
	dataset_path = "./" + DIRECTORY_NAME + "/"
	train_neg_full_path = "train_neg_full.txt"
	train_pos_full_path = "train_pos_full.txt"
	    
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

	# uses too much RAM. I don't know how to fix it so it uses less RAM.
	sorted_train_sentences, indices_of_unique_train_sentences = np.unique(train_sentences, return_index=True)
	train_sentences = train_sentences[indices_of_unique_train_sentences]
	train_labels = train_labels[indices_of_unique_train_sentences]

	np.savetxt(dataset_path + "train_sentences.txt", train_sentences, fmt="%s")
	np.savetxt(dataset_path + "train_labels.txt", train_labels, fmt="%d")
	np.savetxt(dataset_path + "val_sentences.txt", val_sentences, fmt="%s")
	np.savetxt(dataset_path + "val_labels.txt", val_labels, fmt="%d")

options = [
	#stem,  #lemma #spell #remove stopwords
	# [False, False, False, False],
	# [False, False, False, True],
	# [False, True, False, False],
	# [True, False, False, False],
	# [True, True, False, False],
	# [False, True, True, False],
	# [False, False, True, False],
	[True, True, True, True],
]

for option in options:
	SPELLING_CORRECTION = option[2]
	STOPWORD_REMOVAL = option[3]
	STEMMING_CORRECTION = option[0]
	LEMMATIZE_FLAG = option[1]
	directory = ""
	directory += "with-stemming_" if STEMMING_CORRECTION else "no-stemming_"
	directory += "with-lemmatize_" if LEMMATIZE_FLAG else "no-lemmatize_"
	directory += "no-stopwords_" if STOPWORD_REMOVAL else "with-stopwords_"
	directory += "with-spellcorrect" if SPELLING_CORRECTION else "no-spellcorrect"
	# print(directory)
	# print(filtering_strings("I am a dog and I can bark loudly, aggressively, coz i am a doog."))
	# print()
	print(directory)
	DIRECTORY_NAME = directory
	Path("./" + directory + "/").mkdir(parents=True, exist_ok=True)
	preprocess_file("neg")
	print("going to sleep so the CPU can cool down.")
	time.sleep(50)
	print("awoke again, doing next configuration.")
	preprocess_file("pos")
	# do_data_split()
	print("going to sleep so the CPU can cool down.")
	time.sleep(50)
	print("awoke again, doing next configuration.")