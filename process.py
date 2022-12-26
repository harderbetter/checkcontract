# Importing libraries
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

txt1 = []

def remove_string_special_characters(s):
	stripped = re.sub('[^a-zA-z\s]', '', s)
	stripped = re.sub('_', '', stripped)
	stripped = re.sub('\s+', ' ', stripped)
	stripped = stripped.strip()
	if stripped != '':
			return stripped.lower()
		
stop_words = set(stopwords.words('english'))
your_list = []
for i, line in enumerate(txt1):
	txt1[i] = ' '.join([x for
		x in nltk.word_tokenize(line) if
		( x not in stop_words ) and ( x not in your_list )])
	
vectorizer = CountVectorizer(ngram_range =(2, 2))
X1 = vectorizer.fit_transform(txt1)
features = (vectorizer.get_feature_names())


vectorizer = TfidfVectorizer(ngram_range = (2, 2))
X2 = vectorizer.fit_transform(txt1)
scores = (X2.toarray())

sums = X2.sum(axis = 0)
data1 = []
for col, term in enumerate(features):
	data1.append( (term, sums[0, col] ))
ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
words = (ranking.sort_values('rank', ascending = False))

vectorizer = TfidfVectorizer(ngram_range = (3,3))
X2 = vectorizer.fit_transform(txt1)
scores = (X2.toarray())


sums = X2.sum(axis = 0)
data1 = []
for col, term in enumerate(features):
	data1.append( (term, sums[0,col] ))
ranking = pd.DataFrame(data1, columns = ['term','rank'])
words = (ranking.sort_values('rank', ascending = False))

