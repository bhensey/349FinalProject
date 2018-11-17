import nltk
import csv
import random
import string
from nltk.corpus import stopwords

reddit_tokenizer = nltk.TweetTokenizer(strip_handles=True,reduce_len=True)
file = open("redditdata.csv",'r')
#reader = csv.reader(file,delimiter=",(?=([^\']*\"[^\']*\')*[^\']*$)")
all_words = []
data_arr = []

#parse through the csv file
print("parsing through file...")
table = str.maketrans({key: None for key in string.punctuation})
stop_words = set(stopwords.words('english'))
for line in file:
    last_quote = line.rfind(line[0])
    no_quotes = line[last_quote+1:]
    title_string = line[1:last_quote] #get title string
    #title_string = title_string.translate(table) #removes punctuation
    metadata = no_quotes.split(',')
    title = reddit_tokenizer.tokenize(title_string.lower())
    #title = [w for w in title if not w in stop_words] # removes stop words
    score = metadata[1]
    num_comments = metadata[2]
    classification = metadata[3]
    title_word_pairs = [" ".join(pair) for pair in nltk.bigrams(title)]
    #data = title + title_word_pairs #concatenate word pairs
    data = {}
    data['title'] = title
    data['num_comments'] = num_comments
    data['score'] = score
    if not title or not classification:
        print('error occurred')
    data_arr.append((data,classification))
    all_words.extend(data['title'])

all_words_dist = nltk.FreqDist(all_words)
#use up to 2000 words
word_features = list(all_words_dist)[:3000]

def reddit_features(data):
    tokens = data['title']
    word_distribution = nltk.FreqDist(tokens)
    features = {}
    features['num_comments'] = data['num_comments']
    features['score'] = data['score']
    for word in word_features:
        features['contains({})'.format(word)] = word_distribution[word]
    return features

print("size of entire data set is ",len(data_arr))
print("beginning processing data. Number of word features:",len(word_features))
random.shuffle(data_arr)
featuresets = [None] * len(data_arr)
for i in range(len(data_arr)):
    if i % 1000 == 0:
        print(i)
    featuresets[i] = (reddit_features(data_arr[i][0]),data_arr[i][1])
#featuresets = [(reddit_features(f),c) for (f,c) in data_arr]
outfile = open('processed_data.csv','w')
metafile = open('word_metadata.txt','w')

head = list(featuresets[0][0].keys())
ct = 0
for i in range(len(head)):
    if head[i][0:3] == 'con':
        metafile.write(str(ct)+' '+ head[i] +'\n')
        head[i] = "word_" + str(ct)

        ct += 1
outfile.write(','.join(head) + ',class')
outfile.write('\n')
ct = 0
for s in featuresets:
    cls = s[1]
    features = s[0]
    outfile.write(','.join([str(v) for v in features.values()]))
    temp = ',' + cls
    outfile.write(temp)
    #print(len(temp))
    ct += 1
outfile.close()
metafile.close()
print(ct)


######
##test_size = len(featuresets)//10
##print("total number of features:", len(featuresets[0][0]))
##print("Begin cross validation")
##acc = [];
##for i in range(10):
##    testing_fold = featuresets[i*test_size:((i+1)*test_size+1)]
##    training_fold = featuresets[:i*test_size] + featuresets[(i+1)*test_size+1:]
##    classifier = nltk.NaiveBayesClassifier.train(training_fold)
##    accuracy = nltk.classify.accuracy(classifier,testing_fold)
##    acc.append(accuracy)
##    print("cross validation ",i,". Evaluated accuracy: ",
##          accuracy)
##print("average accuracy:", sum(acc)/len(acc));
##classifier.show_most_informative_features(10)
#train_set, test_set = featuresets[test_size:],featuresets[:test_size]
#print("beginning training")
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print("accuracy:",nltk.classify.accuracy(classifier,test_set))
