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
        P_e = np.zeros(len(self.wordIdx))

        # populate with distances between word and each key
        E = list(map(levDist, (word, self.words)))
        P_e = [e * math.log(0.01) - math.log(math.factorial(int(e))) for e in E]
        '''
        for k in range(0, len(wordIdx)):
            E = levDist(word, words[k])
            P_e[k] = E * math.log(0.01) - math.log(math.factorial(int(E)))
        '''
        return P_e
    
    def fixSentence(self, sentenceIn):
        sentenceIn = sentenceIn.split()

        # start of forward message
        n = len(self.wordIdx) 
        f = np.zeros(n)
        sentenceIdx = [self.startIndex]

        # initalizing matrices 
        C = np.zeros([n, len(sentenceIn)])
        D = np.zeros([n, len(sentenceIn)])

        # initalize first values
        D[:,1] = 0 # first column is zeros

        # inital transition matrix (from <s> to character at each row)
        T = np.zeros([n, n])
        for iCurr in range(0, n):
            mask = bigram[:,1] == iCurr + 1
            ip = mask.nonzero()[0]  # the prob dist for after the first character 
            pcurr = bigram[ip,2] # probability distrbution
            iLast = bigram[ip,0].astype(int) - 1 # current word choices

            Tcurr = np.log(np.ones(n) * 1e-6)
            Tcurr[iLast] = pcurr

            T[:,iCurr] = Tcurr

        # distance between first word in sentence and every key word
        E = self.evidenceProb(sentenceIn[0])
        C[:,0] = T[startIndex,:] + E 

        # forward part 
        for j in range(1, len(sentenceIn)):
            Emat = np.zeros(len(words))
            # current word in sentence
            wordCurr = sentenceIn[j]
            
            for i in range(0, n):
                # evidence matrix between current word in sentence and current key
                E = levDist(wordCurr, words[i])
                #E = E * math.log(0.01) - math.log(math.factorial(int(E)))
                Emat[i] = 100*E

                C[i,j] = np.max(C[:,j-1] + T[:,i] + E)
                D[i,j] = np.argmax(C[:,j-1] + T[:,i] + E)
                
                #C[i,j] = np.max(C[:,j-1]  + 100*E)
                #D[i,j] = np.argmax(C[:,j-1] + 100*E)

        # Backward Part
        sentenceSmoothed = []
        bestIdx = np.argmax(C[:,-1])
        prob = 10**np.max(C[:,-1])

        for h in range(len(sentenceIn)-1, -1, -1):
            sentenceSmoothed.append(words[int(bestIdx)])
            bestIdx = D[int(bestIdx),h]

        sentenceSmoothedStr = ""
        for i in range(len(sentenceSmoothed)-1, -1, -1):
            sentenceSmoothedStr = sentenceSmoothedStr + (sentenceSmoothed[i] + " ")