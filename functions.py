# all functions used in the sentence generation scheme
import random 
import math
import numpy as np 
import matplotlib.pyplot as plt
from LevDistance import levDist

class NGramModel:
    def __init__(self, vocabFile, unigramFile, bigramFile, trigramFile):
        self.vocab = self.load_vocab(vocabFile)
        self.unigram = self.load_unigram(unigramFile)
        self.bigram = self.load_bigram(bigramFile)
        self.trigram = self.load_trigram(trigramFile)

    def load_vocab(self, vocab_file):
        # importing txt files 
        with open(vocab_file, "r") as file:
            vocabRaw = file.read()

        # split string in txt 
        vocabRaw = vocabRaw.split()
        vocab = []
        for i in range(0, len(vocabRaw), 2):
            vocab.append((vocabRaw[i], vocabRaw[i+1]))
        return vocab
    
    def load_unigram(self, unigram_file):
        # unigram
        with open(unigram_file, "r") as file:
            unigramRaw = file.read()

        # split string in txt 
        unigramRaw = unigramRaw.split()
        unigram = []
        for i in range(0, len(unigramRaw), 2):
            unigram.append((int(unigramRaw[i]), 10**float(unigramRaw[i+1])))
        return unigram
    
    def load_bigram(self, bigram_file):
        # bigram
        with open("data/bigram_counts.txt", "r") as file:
            bigramRaw = file.read()

        # split string in txt 
        bigramRaw = bigramRaw.split()
        bigram = []
        for i in range(0, len(bigramRaw), 3):
            bigram.append((int(bigramRaw[i]), int(bigramRaw[i+1]), 10**float(bigramRaw[i+2])))
        bigram = np.array(bigram)
        return bigram

    def load_trigram(self, trigram_file):
        # trigram
        with open(trigram_file, "r") as file:
            trigramRaw = file.read()

        # split string in txt 
        trigramRaw = trigramRaw.split()
        trigram = []
        for i in range(0, len(trigramRaw), 4):
            trigram.append((int(trigramRaw[i]), int(trigramRaw[i+1]), int(trigramRaw[i+2]), 10**float(trigramRaw[i+3])))
        trigram = np.array(trigram)
        return trigram
    
    def useUnigram(self):
        # start generating sentence (n=1, bigram)
        pcurr = self.unigram[:,1] # probability distrbution
        icurr = self.unigram[:,0] # current word choices
        return icurr, pcurr

    def useBigram(self, last):
        # start generating sentence (n=1, bigram)
        mask = self.bigram[:,0] == last + 1
        ip = mask.nonzero()[0]  # the prob dist for after the first character
        if ip.size == 0:
            return None, None 
        pcurr = self.bigram[ip,2] # probability distrbution of word after word_last
        icurr = self.bigram[ip,1] # word indices coming after word_last
        return icurr, pcurr # returns the probability distribution of all the words accossiated with the word_last

    def useTrigram(self, lastlast, last): # if we have word_lastlast and word_last, what is the probability distrubution for word?
        # start generating sentence (n=>2, trigram)
        try:
            a = (self.trigram[:,0] == lastlast + 1)
            b = (self.trigram[:,1] == last + 1)
            ip = np.logical_and(a,b).nonzero()[0] # find where word last last and word last 
        except: 
            return None, None
        if ip.size == 0:
            return None, None # this combination is not in the trigram
        icurr = self.trigram[ip,2] # current word choices
        pcurr = self.trigram[ip,3] # probability distrbution
        return icurr, pcurr # probability of words coming after word_lastlast and word_last from the trigram

class sentenceMaker:
    def __init__(self, Ngram):
        self.Ngram = Ngram
        self.wordIndex = [v[0] for v in Ngram.vocab]
        self.words = [v[1] for v in Ngram.vocab]

        self.startIndex = self.words.index("<s>")

        # initalizing the last and second last words in the sentence
        self.last = None 
        self.lastlast = None
    
    def plotProbDist(self, icurr, pcurr, iword):
        # plot probability distrbution
        fig, ax = plt.subplots()
        w = np.array(self.words)
        words = w[np.array(icurr-1,dtype=np.int16)]
        ax.bar(words, pcurr)

        try:
            ax.set_title("Prob Dist after: " + self.words[self.lastlast] + " " + self.words[self.last])
        except:
            ax.set_title("Prob Dist after: " + self.words[self.last])
        
        ax.set_xticks(range(len(words)))  # Set positions of ticks
        ax.set_xticklabels(words, rotation=45, ha='right')  # Rotate labels

        # Adjust layout to fit rotated labels
        fig.tight_layout()
        plt.show()

    def generate_sentence(self, show = 0):
        # starting sentence character must always be the first 'word'
        startIndex = self.words.index("<s>")
        sentence = []
        sentence.append(self.words[startIndex] + " ")

        # update last word
        self.last = startIndex
        
        distUse = []
        wordNew = []

        count = 1 
        while (wordNew != "</s>") and (count < 50): # stop when end line character reached
            distUsed = None
            # use trigram
            icurr, pcurr = self.Ngram.useTrigram(self.lastlast, self.last)
            distCurr = "Trigram"

            if icurr is None: # if this combo doesn't exist in trigram, use bigram
                icurr, pcurr = self.Ngram.useBigram(self.last)
                distCurr = "Bigram"

                if icurr is None: # if this combo doesn't exist in either bi or trigram, sample a single word
                    icurr, pcurr = self.NgramuseUnigram()
                    distCurr = "Unigram"

            distUse.append(distCurr + ", ")


            # sample a word based on the weights
            i = random.choices(icurr, weights = pcurr)
            iword = int(i[0]) - 1

            # plot the probability distrubution
            if show:
                self.plotProbDist(icurr, pcurr, iword)

            wordNew = self.words[iword]
            sentence.append(wordNew)

            # add a space between words
            sentence.append(" ")

            # update indices 
            self.lastlast = self.last
            self.last = iword

            count += 1
        return "".join(sentence), "".join(distUse)
    
    def evidenceProb(self, word):
        # find evidence prob matrix for the current word and all keys 

        # populate with distances between word and each key
        E = list(map(levDist, [word]*len(self.words), self.words))
        P_e = [e * math.log10(0.01) - math.log10(math.factorial(int(e))) for e in E]
        return P_e
    
    def fixSentence(self, sentenceIn):
        # implementing Viterbi algorithm to update probabilities of each word at each location in the 
        # sentence sequence 
        sentenceIn = sentenceIn.split()

        # start of forward message
        n = len(self.wordIndex) # number of vocab words

        # initalizing matrices 
        P = np.zeros([n, len(sentenceIn)]) # the max Probability of ending up on each word in the sentence sequence
        P_i = np.zeros([n, len(sentenceIn)]) # tracks the index of the previous state (word) leading to the current word in the current sequence 

        # initalize first values
        P_i[:,1] = 0 # first column is zeros

        # inital transition matrix (from <s> to character at each row)
        T = np.zeros([n, n])
        for iCurr in range(0, n):
            mask = self.Ngram.bigram[:,1] == iCurr + 1
            ip = mask.nonzero()[0]  # the prob dist for after the first character 
            pcurr = self.Ngram.bigram[ip,2] # probability distrbution from <s> to all other characters
            iLast = self.Ngram.bigram[ip,0].astype(int) - 1 # all words preceeding <s>

            # transition probabilities from <s> to all other words initalized to ~zero, then take log of prob 
            Tcurr = np.log10(np.ones(n) * 1e-6)
            Tcurr[iLast] = np.log10(pcurr) # update known transition probabilities

            T[:,iCurr] = Tcurr # update total transition matrix

        # distance between first word in sentence and every vocabulary word
        E = self.evidenceProb(sentenceIn[0])
        P[:,0] = T[self.startIndex,:] + E 

        # forward part 
        for j in range(1, len(sentenceIn)):
            # current word in sentence
            wordCurr = sentenceIn[j]
            
            # proability the the current word is the vocab word (if d is small, higher likelyhood it is the word)
            E = self.evidenceProb(wordCurr)
            
            for i in range(0, n):
                # since we are working in log(prob), we can add the previous state probability, transition prob, and prob the vocab word is the misspelled word 
                P[i,j] = np.max(P[:,j-1] + T[:,i] + E[i])
                P_i[i,j] = np.argmax(P[:,j-1] + T[:,i] + E[i])
                

        # Backward Part
        sentenceSmoothed = []
        bestIdx = np.argmax(P[:,-1])
        prob = 10**np.max(P[:,-1])

        for h in range(len(sentenceIn)-1, -1, -1):
            sentenceSmoothed.append(self.words[int(bestIdx)]) # the most likely word (state) in that sequence
            bestIdx = P_i[int(bestIdx), h] # the word (state) that lead to the current word (state)

            sentenceSmoothed.append(" ")
        sentenceSmoothed.reverse() 
        return "".join(sentenceSmoothed)