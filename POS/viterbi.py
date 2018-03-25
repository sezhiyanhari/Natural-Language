import sys,re
from itertools import izip
from collections import defaultdict
from sets import Set
import math

vocab={}
OOV_WORD="OOV"
HMM_FILE=sys.argv[1]
TEST_FILE=sys.argv[2]
transitions=defaultdict(float)
emissions=defaultdict(float)
policy=defaultdict(float)
backptr=defaultdict(str)
K = Set([])

with open(HMM_FILE) as hmmFile, open(TEST_FILE) as testFile:

	for hmmString in hmmFile:

		hmmData=re.split("\s+", hmmString.rstrip())

		# data is structured like transitions(tag_i_2, tag_i_1, tag): probability
		# represents (tag | tag_i-1, tag_i_2) probability
		if (hmmData[0] == "trans"):
			transitions[(hmmData[1], hmmData[2], hmmData[3])] = hmmData[4]

		# data is structured like emissions(tag, word): probability
		# represents (word | tag) probability
		if (hmmData[0] == "emit"):
			emissions[(hmmData[1], hmmData[2])] = hmmData[3]
			if (hmmData[2] not in vocab):
				vocab[(hmmData[2])] = 1
				
		# set containing all possible
		if (hmmData[1] not in K):
			K.add(hmmData[1])
		K.add("final")

	# should be the crux of the code	
	for line in testFile:
		line = line.strip()
		words = line.split() # every sentences
		k_sets = [None] * (len(words) + 2)
		output_string = [str] * (len(words))

		for sets in range(len(words) + 2):
			if (sets == 0 or sets == 1):
				k_sets[sets] = Set(["init"])
			else:
				k_sets[sets] = K

		# k = 0, initialize to 1 (in code implementation, policy index starts from 1)
		policy[(1, "init", "init")] = 0.0

		# k = 1, 2, 3 ... n (on formula sheet, but 2, 3, ... n + 1 in code)
		for k_iter in xrange(2, len(words) + 2):
			# the sentence indices are offset by 2 to accomodate the set indices
			currentWord = (words[k_iter - 2])
			if (currentWord not in vocab):
				currentWord = OOV_WORD[:]
			for v in k_sets[k_iter]:
				for u in k_sets[k_iter - 1]:
					maxPolicy = float("-inf")
					emptyTag = "EMPTY"
					for w in k_sets[k_iter - 2]:
						if (float(transitions[(w, u, v)]) == 0.0):
							transitions[(w, u, v)] = -99999
						if (float(emissions[(v, currentWord)]) == 0.0):
							emissions[(v, currentWord)] = -99999
						newPolicy = policy[(k_iter - 1, w, u)] + float(transitions[(w, u, v)]) + float(emissions[(v, currentWord)])
						if (newPolicy > maxPolicy):
							maxPolicy = newPolicy
							emptyTag = w[:]
					policy[(k_iter, u, v)] = maxPolicy
					backptr[(k_iter, u, v)] = emptyTag[:]

		# produce tag sequence here
		# n = len(words) + 1
		# below here has been debugged properly
		n = len(words) + 1
		y_n_1 = "EMPTY"
		y_n = "EMPTY"
		lastTagsMax = float("-inf")
		for v in k_sets[n]:
			for u in k_sets[n - 1]:
				if (float(transitions[(u, v, "final")]) == 0):
					transitions[(u, v, "final")] = -99999
				newLastTagsPolicy = policy[(n, u, v)] + float(transitions[(u, v, "final")])
				if (newLastTagsPolicy > lastTagsMax):
					lastTagsMax = newLastTagsPolicy
					y_n_1 = u[:]
					y_n = v[:]

		output_string[n - 2] = y_n
		output_string[n - 3] = y_n_1

		for k_iter in xrange(n - 4, -1, -1):
			output_string[k_iter] = backptr[(k_iter + 4, output_string[k_iter + 1], output_string[k_iter + 2])][:]
		
		print(" ".join(output_string))

		policy.clear()
		backptr.clear()

