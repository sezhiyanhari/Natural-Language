import math, collections
# custom model using trigram with bigram and unigram backoff
class CustomModel:

  def __init__(self, corpus):
    self.smoothTrigramCounts = collections.defaultdict(lambda: 0)
    self.smoothBigramCounts = collections.defaultdict(lambda: 0)
    self.smoothUnigramCounts = collections.defaultdict(lambda: 0)
    self.vocab = 0
    self.total = 0
    self.train(corpus)

  def train(self, corpus):

    word_i_2 = ''
    word_i_1 = ''
    word_i = ''
    # TODO your code here
    counter = 0
    for sentence in corpus.corpus:
      for datum in sentence.data:
        current_word = datum.word
        if(counter == 0):
          word_i_2 = current_word[:]
        if(counter == 1):
          word_i_1 = current_word[:]
          self.smoothBigramCounts[(word_i_1, word_i_2)] = self.smoothBigramCounts[(word_i_1, word_i_2)] + 1 # dealing with initial bigrams
        if(counter >= 2):
          word_i = current_word[:]
          self.smoothTrigramCounts[(word_i, word_i_1, word_i_2)] = self.smoothTrigramCounts[(word_i, word_i_1, word_i_2)] + 1
          self.smoothBigramCounts[(word_i_1, word_i_2)] = self.smoothBigramCounts[(word_i_1, word_i_2)] + 1
          word_i_2 = word_i_1[:]
          word_i_1 = current_word[:]

        self.smoothUnigramCounts[current_word] = self.smoothUnigramCounts[current_word] + 1 # dealing with all unigram counts
        if(self.smoothUnigramCounts[current_word] == 1):
            self.vocab += 1
        self.total += 1
        counter += 1

      self.smoothBigramCounts[(word_i_1, word_i_2)] = self.smoothBigramCounts[(word_i_1, word_i_2)] + 1 # dealing with final bigram counts
      counter = 0

  def unigram_increment(self, word_i): # smoothed unigram
    count = self.smoothUnigramCounts[word_i] + 1
    increment = math.log(count) - math.log(self.total + self.vocab)

    return increment

  def bigram_increment(self, word_i, word_i_1): # smoothed bigram
    numerator = (self.smoothBigramCounts[(word_i, word_i_1)])
    denominator = self.smoothUnigramCounts[word_i_1]
    increment = math.log(numerator) - math.log(denominator)

    return increment

  def trigram_increment(self, word_i, word_i_1, word_i_2): # unsmoothed trigram
    numerator = (self.smoothTrigramCounts[(word_i, word_i_1, word_i_2)])
    denominator = self.smoothBigramCounts[(word_i, word_i_1)]
    increment = math.log(numerator) - math.log(denominator)

    return increment
 
  def score(self, sentence):
    
    score = 0.0
    word_i_2 = sentence[0]
    word_i_1 = sentence[1]
    for word_i in sentence[2:len(sentence)]:
      if(self.smoothTrigramCounts[(word_i, word_i_1, word_i_2)] > 0):
        score = score + self.trigram_increment(word_i, word_i_1, word_i_2)
      elif(self.smoothBigramCounts[(word_i, word_i_1)] > 0):
        score = score + self.bigram_increment(word_i, word_i_1)
      else:
        score = score + self.unigram_increment(word_i)
      word_i_2 = word_i_1[:]
      word_i_1 = word_i[:]
      
    return score
