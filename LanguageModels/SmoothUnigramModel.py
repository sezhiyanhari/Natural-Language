import math, collections

class SmoothUnigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothUnigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.vocab = 0
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

    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.smoothUnigramCounts[token] = self.smoothUnigramCounts[token] + 1 # determines number of tokens, grouped
        self.total += 1 # counts number of total tokens (all words, not unique words)
        if(self.smoothUnigramCounts[token] == 1):
            self.vocab += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    for token in sentence:
      count = self.smoothUnigramCounts[token] + 1
      if count > 0:
        score += math.log(count)
        score -= math.log(self.total + self.vocab)
      # Ignore unsee


    return score
