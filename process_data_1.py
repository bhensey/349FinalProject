import nltk
import csv
import random
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

STEMMER = SnowballStemmer("english")
TOKENIZER = nltk.TweetTokenizer(strip_handles=True,reduce_len=True)
STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = str.maketrans({key: None for key in string.punctuation})  # remove punctuation


def process_data(inputFile, metaFile, outputFile, num_features):

	def tokenized_dist(title):
		'''
		Takes a title string and converts to tokens given the following rules:
		1) split by word
		'''
		#title = title.translate(PUNCTUATION)
		tokens = TOKENIZER.tokenize(title.lower())
		#tokens = [w for w in tokens if not w in STOPWORDS]
		tokens = [STEMMER.stem(token) for token in tokens]
		#bigram_tokens = [" ".join(pair) for pair in nltk.bigrams(tokens)]
		#tokens = bigram_tokens + tokens
		dist = nltk.FreqDist(tokens)
		return dist

	def getData(line):
		'''
		Get title and subreddit from raw input line
		'''
		comma_index = line.rfind(',')
		line_subreddit = line[comma_index+1:].rstrip()
		line_title = line[1:comma_index-1]
		return line_title, line_subreddit

	# Determine title features
	file = open(inputFile,'r')
	word_dist = nltk.FreqDist()  
	processed_data = []
	print("parsing through file...")
	for line in file:
		line_title, line_subreddit = getData(line)
		line_title_dist = tokenized_dist(line_title)
		word_dist += line_title_dist
		processed_data.append({'title':line_title,'tokens':line_title_dist,'subreddit':line_subreddit})

	file.close()
	word_features_raw = word_dist.most_common(num_features)
	word_features = [word[0] for word in word_features_raw]
	with open(metaFile, 'w') as file:
		for word in word_features_raw:
			file.write(word[0].ljust(15) + str(word[1])+'\n')


	# Vectorize data
	random.shuffle(processed_data)
	outFile = open(outputFile,'w')
	for i in range(num_features):
		outFile.write('word_'+str(i)+',')
	outFile.write('class\n') 
	for data in processed_data:
		for word in word_features:
			outFile.write(str(data['tokens'][word] > 0) + ',') #boolean values
		outFile.write(data['subreddit']+'\n')
	outFile.close()
		
process_data("redditData.csv", 'metadata.csv','processed_data.csv', 5000)


	   