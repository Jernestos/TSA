Preprocessing - READMe
This READMe describes how to use the files contained in this folder.
-----------------------
Content: This folder contains
- a python file "preprocess.py"
- a python file "remove_dups_and_split.py"
- a colab file "minimal_preprocessing.ipynb"
-----------------------
Prerequisites (P):
Tools:
	Before running the code, please install the following (if not already done) via pip:

	pip install num2words
	pip install symspellpy
	pip install re
	pip install nltk

	or alternativly using pip3:

	pip3 install num2words
	pip3 install symspellpy
	pip3 install re
	pip3 install nltk

Files:
This folder is located in ./
The twitter datasets files have to be in to the folder ./raw/
-----------------------
-----------------------

"preprocess.py":
-----------------------
Prerequisites:
- Prerequisites described above (P)

- User action:
There is a variable "options" that stores a list of lists. Each list encodes if stemming, lemmatizatization, stopword removal and spell correction is to be performed on the tweets to be read in. The user can manually select (via commenting out/not commenting out lists) what the preprocessing step should consists of.
-----------------------
How to run "preprocess.py":
python3 preprocess.py
-----------------------
Output:
After running this file, and depending on the user's choices of elements for preprocessing step (See Prerequisites, User action), there are folder appropriately named in ./     .
These subfolders contain the user-specified-preprocessed (see Prerequisites, User action) tweets, still seperated by positive and negative labels. 
-----------------------
-----------------------

"remove_dups_and_split.py":
-----------------------
Prerequisites:
- Prerequisites described above (P)
- python file "preprocess.py" must have been run beforehand
-----------------------
How to run "preprocess.py":
python3 remove_dups_and_split.py
-----------------------
Output:
For each of the subfolders, created by running python file "preprocess.py" (except one case *), the positive and negative tweets are combined in a set, duplicates are removed it, from which training and validation dataset is created.

Exception case (*):
By raw, no preprocessing is done on the files.

"minimal_preprocessing.ipynb":
-----------------------
How to run "preprocess.py":
run all cells inside te file
-----------------------
Output:
Outputs minimal preprocessing files.

