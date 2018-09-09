#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

# maps words to their counts
word2count = {}
prev_label = None
# input comes from STDIN
for line in sys.stdin:
	# remove leading and trailing whitespace
	line = line.strip()

	# parse the input we got from mapper.py
	current_label,word,count = line.split('\t')
	if ( prev_label == None):
		prev_label = current_label
	if ( prev_label == current_label ):
	# convert count (currently a string) to int
		try:
			count = int(count)
		except ValueError:
       			# count was not a number, so silently
       			# ignore/discard this line
			continue
		try:
			word2count[word] = word2count[word] + count
   		except:
			word2count[word] = count
		prev_label = current_label	
	else:
		for word in word2count.keys():
			print('%s\t%s\t%s' %(prev_label,word,word2count[word] ))
		word2count = {}
		prev_label = current_label




