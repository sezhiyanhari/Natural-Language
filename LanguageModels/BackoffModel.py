import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothBigramCounts = collections.defaultdict(lambda: 0)
    self.smoothUnigramCounts = collections.defaultdict(lambda: 0)
    self.uniqueCounts = collections.defaultdict(lambda: 0)
    self.vocab = 0
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    previous_word = '<s>'
    skip_counter = 0

    for sentence in corpus.corpus:
      for datum in sentence.data:
        # Smooth Bigram Model
        current_word = datum.word
        if(previous_word != '<s>' or skip_counter == 1): # not sure if "and current_word != "</s>"
            self.smoothBigramCounts[(current_word, previous_word)] = self.smoothBigramCounts[(current_word, previous_word)] + 1 # determines number of tokens, grouped
            self.uniqueCounts[previous_word] = self.uniqueCounts[previous_word] + 1
            previous_word = current_word[:]
        else:
            skip_counter = 1  

        # Smooth Unigram Model
        self.smoothUnigramCounts[current_word] = self.smoothUnigramCounts[current_word] + 1 # determines number of tokens, grouped
        if(self.smoothUnigramCounts[current_word] == 1):
            self.vocab += 1
        self.total += 1
      skip_counter = 0

  def unigram_increment(self, token):
    count = self.smoothUnigramCounts[token] + 1
    increment = math.log(count) - math.log(self.total + self.vocab)

    return 0.8 * increment
  
  def bigram_increment(self, current_word, previous_word):
    numerator = (self.smoothBigramCounts[(current_word, previous_word)])
    denominator = self.uniqueCounts[previous_word]
    increment = math.log(numerator) - math.log(denominator)

    return increment

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    previous_word = sentence[0]
    score = 0.0 
    for current_word in sentence:
        if((self.smoothBigramCounts[(current_word, previous_word)]) > 0):
            score = score + self.bigram_increment(current_word, previous_word)
        else:
            score = score + self.unigram_increment(current_word)
        previous_word = current_word[:]
    
    return score

