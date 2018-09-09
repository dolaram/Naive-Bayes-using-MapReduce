#!/usr/bin/env python
"""mapper.py"""

import sys
import string
stopwords = ['i','me','my','myself','we','our','ours','ourselves','you',"you're",
 "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
 'his', 'himself', 'she',"she's",'her','hers', 'herself', 'it', "it's", 'its', 'itself',
 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
 'be', 'been', 'being', 'have',
 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
 'between', 'into', 'through', 'during', 'before',
 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
 "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
# input comes from STDIN (standard input)
for line in sys.stdin:
	# remove leading and trailing whitespace  
        line = line.strip()

        #  split the line into words 
        labels, sentence = line.split('\t',1)
	
	# one sample may belong to multiple classes so add samples to all those classes
	labels = labels.strip().split(',')
	
	# remove the url from sample
	sentence_split = sentence.split()
	sentence = ' '.join([x.lower() for x in sentence_split if ('<' not in x) and ('\\' not in x)])
	# remove the punctuation from the 
	text = ''.join([char for char in sentence if char not in string.punctuation])
	
	# split the text into individual word
	text_split = text.split()
	
	# remove the stop words
	words = [w for w in text_split if w not in stopwords]
	
	# out to STDN output
	for label in labels:
		for word in words:
			print('%s\t%s\t%s' %(label,word,1))

        """ CODE ENDS HERE """

        """for word in words:
		for label in labels:
	                print('%s\t%s\t%s' %(label,word,1))  """
	# normal program ends here

	"""	
	label_list, text = line.split('\t',1)
	labels = label_list.split(',')
	words = text.split()
	for w in words:
		for label in labels:
			print('%s\t%s\t'%(label,w,1))
	"""

	#words = text.split()
	#remove leading and trailing whitespace  
	#line = line.strip()
	# split the line into words
	#words = text.split()
	# split the line into word in words
	#for w in words:
	#	for label in labels:
	#		print('%s\t%s\t%s', %(label.strip(),w.strip(),1))
	#label = words[0]
	#for word in words:
	#	print('%s\t%s\t%s' %(label,word,1))

