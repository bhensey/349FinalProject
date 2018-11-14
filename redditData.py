#!/usr/bin/env python3
import praw

f = open("redditData.arff", "w+")
subredditList = {"AskReddit":"1", "politics":"2", "worldnews":"3", "nba":"4", "funny":"5", "movies":"6"}
reddit = praw.Reddit(client_id='qr7w0xIMBNVwvQ', \
                 client_secret='UBFVrgbyVWG-5_drnRX9uqkDv9w', \
                 user_agent='EECS349', \
                 username='EECS349', \
                 password='uF*TAV6mYis&')

f.write("@relation training" + '\n\n')
f.write("@attribute 'title' string" + '\n')
f.write("@attribute 'score' numeric" + '\n')
f.write("@attribute 'num_comments' numeric" + '\n')
f.write("@attribute 'subreddit' string" + '\n\n')
f.write("@data" + '\n')

# Write data from top 5 posts for each subreddit
for subreddit in subredditList:
	subredditObj = reddit.subreddit(subreddit)
	for submission in subredditObj.top(limit=1000):
		data = [str(submission.title.encode('ascii', 'ignore')), str(submission.score), str(submission.num_comments), subreddit]

		f.write(','.join(data)[1:] + '\n')
		#f.write((str(u','.join(data).encode())) + '\n')
f.close()

'''
# Initializing data file as redditData.txt
f = open("redditData.txt", "w+")
f.write("")

'''