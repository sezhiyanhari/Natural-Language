#!/usr/bin/python

# David Bamman
# 2/14/14
#
# Python port of train_hmm.pl:

# Noah A. Smith
# 2/21/08
# Code for maximum likelihood estimation of a bigram HMM from 
# column-formatted training data.

# Usage:  train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are 
# implied.  
# The output format is the HMM file format as described in viterbi.pl.

import sys,re
from itertools import izip
from collections import defaultdict
import math

TAG_FILE=sys.argv[1]
TOKEN_FILE=sys.argv[2]

vocab={}
OOV_WORD="OOV"
INIT_STATE1="init"
INIT_STATE2="init"
FINAL_STATE="final"

emissions={}
transitions={}
transitionsTrigram={}
transitionsTotal=defaultdict(int)
transitionsTrigramTotal=defaultdict(int)
emissionsTotal=defaultdict(int)

lambda_1 = 0
lambda_2 = 0
lambda_3 = 0
N = 0

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
	for tagString, tokenString in izip(tagFile, tokenFile):
		tags=re.split("\s+", tagString.rstrip())
		tokens=re.split("\s+", tokenString.rstrip())
		pairs=zip(tags, tokens)
		tag_i_2=INIT_STATE2
		tag_i_1=INIT_STATE1
		for (tag, token) in pairs:
			# this block is a little trick to help with out-of-vocabulary (OOV)
			# words.  the first time we see any word token, we pretend it
			# is an OOV.  this lets our model decide the rate at which new
			# words of each POS-type should be expected (e.g., high for nouns,
			# low for determiners).
			if token not in vocab:
				vocab[token]=1
				token=OOV_WORD

			if tag not in emissions:
				emissions[tag]=defaultdict(int)
			if tag_i_1 not in transitions:
				transitions[tag_i_1]=defaultdict(int)
			if (tag_i_2, tag_i_1) not in transitionsTrigram:
				transitionsTrigram[(tag_i_2, tag_i_1)]=defaultdict(int)

			# increment the emission/transition observation
			emissions[tag][token]+=1
			emissionsTotal[tag]+=1
			
			transitions[tag_i_1][tag]+=1
			transitionsTotal[tag_i_1]+=1

			transitionsTrigram[(tag_i_2, tag_i_1)][tag]+=1
			transitionsTrigramTotal[(tag_i_2, tag_i_1)]+=1

			N += 1.0

			if (transitionsTrigram[(tag_i_2, tag_i_1)][tag] > 0):
				case1 = -999
				case2 = -999
				case3 = -999

				if (transitionsTrigramTotal[(tag_i_2, tag_i_1)] != 1):
					case1 = 1.0 * (transitionsTrigram[(tag_i_2, tag_i_1)][tag] - 1) / (transitionsTrigramTotal[(tag_i_2, tag_i_1)] - 1)
				if (transitionsTotal[tag_i_1] != 1):
					case2 = 1.0 * (transitionsTrigramTotal[(tag_i_1, tag)] - 1) / (transitionsTotal[tag_i_1] - 1)
				if (N != 1):
					case3 = 1.0 * (transitionsTotal[tag] - 1) / (N - 1)

				if (case1 >= case2 and case1 >= case3):
					lambda_3 += transitionsTrigram[(tag_i_2, tag_i_1)][tag]
				elif (case2 >= case3):
					lambda_2 += transitionsTrigram[(tag_i_2, tag_i_1)][tag]
				else:
					lambda_1 += transitionsTrigram[(tag_i_2, tag_i_1)][tag]
			
			tag_i_2 = tag_i_1[:]
			tag_i_1 = tag[:]

		# don't forget the stop probability for each sentence
		if tag_i_1 not in transitions:
			transitions[tag_i_1]=defaultdict(int)

		if (tag_i_2, tag_i_1) not in transitionsTrigram:
			transitionsTrigram[(tag_i_2, tag_i_1)]=defaultdict(int)

		transitions[tag_i_1][FINAL_STATE]+=1
		transitionsTotal[tag_i_1]+=1
		transitionsTotal[FINAL_STATE]+=1

		transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE]+=1
		transitionsTrigramTotal[(tag_i_2, tag_i_1)]+=1
		transitionsTrigramTotal[(tag_i_1, FINAL_STATE)]+=1

		N+=1

		if (transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE] > 0):
			case1 = -999
			case2 = -999
			case3 = -999

			if (transitionsTrigramTotal[(tag_i_2, tag_i_1)] != 1):
				case1 = 1.0 * (transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE] - 1) / (transitionsTrigramTotal[(tag_i_2, tag_i_1)] - 1)
			if (transitionsTotal[tag_i_1] != 1):
				case2 = 1.0 * (transitionsTrigramTotal[(tag_i_1, FINAL_STATE)] - 1) / (transitionsTotal[tag_i_1] - 1)
			if (N != 1):
				case3 = 1.0 * (transitionsTotal[FINAL_STATE] - 1) / (N - 1)

			if (case1 >= case2 and case1 >= case3):
				lambda_3 += transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE]
			elif (case2 >= case3):
				lambda_2 += transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE]
			else:
				lambda_1 += transitionsTrigram[(tag_i_2, tag_i_1)][FINAL_STATE]

total_lambda = lambda_3 + lambda_2 + lambda_1
lambda_3 = 1.0 * lambda_3 / total_lambda
lambda_2 = 1.0 * lambda_2 / total_lambda
lambda_1 = 1.0 * lambda_1 / total_lambda


for tag_i_2, tag_i_1 in transitionsTrigram:
	for tag in transitionsTrigram[(tag_i_2, tag_i_1)]:
		score = 0.0
		score = score + float(transitionsTrigram[(tag_i_2, tag_i_1)][tag]) / transitionsTrigramTotal[(tag_i_2, tag_i_1)] * lambda_3
		score = score + float(transitionsTrigramTotal[(tag_i_1, tag)]) / transitionsTotal[tag_i_1] * lambda_2
		score = score + float(transitionsTotal[tag]) / (N - 1) * lambda_1
		# final_score = math.log(score)
		final_score = score
		print "trans %s %s %s %f" % (tag_i_2, tag_i_1, tag, final_score)
"""
for tag in emissions:
	for token in emissions[tag]:
		print "emit %s %s %f" % (tag, token, math.log(float(emissions[tag][token]) / emissionsTotal[tag]))
"""
