import nltk
import csv
import random

reddit_tokenizer = nltk.TweetTokenizer(strip_handles=True,reduce_len=True)
file = open("redditdata.csv",'r')
#reader = csv.reader(file,delimiter=",(?=([^\']*\"[^\']*\')*[^\']*$)")
all_words = []
data_arr = []

#parse through the csv file
print("parsing through file...")
for line in file:
    last_quote = line.rfind(line[0])
    no_quotes = line[last_quote+1:]
    title_string = line[1:last_quote]
    #print(no_quotes)
    metadata = no_quotes.split(',')
    title = reddit_tokenizer.tokenize(title_string.lower())
    #print(metadata)
    score = metadata[1]
    num_comments = metadata[2]
    classification = metadata[3]
    data_arr.append((title,classification))
    all_words.extend(title)

all_words_dist = nltk.FreqDist(all_words)
#use up to 1000 words
word_features = list(all_words_dist)[:3000]

def reddit_features(title_tokens):
    word_distribution = nltk.FreqDist(title_tokens)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = word_distribution[word]
    return features

print("size of entire data set is ",len(data_arr))
print("beginning processing data...")
random.shuffle(data_arr)
featuresets = [None] * len(data_arr)
for i in range(len(data_arr)):
    if i % 1000 == 0:
        print(i)
    featuresets[i] = (reddit_features(data_arr[i][0]),data_arr[i][1])
#featuresets = [(reddit_features(f),c) for (f,c) in data_arr]
test_size = len(featuresets)//10
train_set, test_set = featuresets[test_size:],featuresets[:test_size]
print("beginning training")
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy:",nltk.classify.accuracy(classifier,test_set))
