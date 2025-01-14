This script will (A) generate text or continue writing a sentence and (B) correct the words in a sentence using Markov and hidden Markov models to piece together the probabilities of preceding and succeeding words in a sentence based on known probability distributions.

The 2 data files available are: 

1) all available vocabulary in the file "vocab.txt", numbered from 1 to n (number of words):

1 word1
2 word2
...
n wordn

If we have a sentence so that there is a word at each time point t (x_t), in the order: x_0 -> x_1 -> x_2 -> x_3 -> x_4  
2) a file stating the unitary probability of word with integer index i occuring.: 

i log( P( x_t = i ) )

3) a file stating the binary probability of word j occuring after word i:

i j log( P( x_t = j | x_t-1 = i) )

4) a file stating the trinary probability of word k occuring when preceeded by word j and i:

i j k log( P( x_t = k | x_t-1 = j, x_t-2 = i ) ) 


TASKS: 
Both tasks are found in the main.py file.

(A) Sentence Generation: 
The sentence is generated using a Markov model of the Bayesian Network by sampling from the probability P(x_t = k | x_t-1 = j, x_t-2 = i)

For the first word, the sentence start will always be the character "<s>", treated as a word in the vocab.txt
The words will be generated the prior sample: 
                                P(x_1...x_n) = product_i=1 to n ( P (x_i | parents(x_i) ) 
sentences end when the "</s> character is reached.
The transition probability from each state (word_t) to another state (word_t+1) is first given by the trigram using word_t-1, if
the combination is not available, the transition probability will back off to the bigram, then finally the unigram.

(B) Sentence Correction
A first order hidden Markov model as depicted bellow is model the true sentence sequence given an incorrect input sentence.

X_0 ---> X_1 ---> X_2 ---> X_3 ---> X_4 ...
          |        |        |        |
          v        v        v        v
         E_1      E_2      E_3      E_4

Firstly, only words from the avilable vocabulary can be in the sentence. The probability of an input word (u) being a vocabilary 
word (v) is 
                                P(E_t = u | X_t = v) = (L^k) * exp(-L) / k! 
                                log(P) = klog(L) - log(k!) + c                              [log of probability]
where k is the Levenshtein distance between u and v, and L is a constanct 0.01.

Let j represent the word index for each known word in the vocabilary, and t be the location in the sentence sequence.
i) Each state (X_t,j) in the sequence (t) has a probability of being a word (j) E_t,j [EE(X_t -> E_t,j)]
ii) and a transition probility from state X_t-1,j [T(X_t-1 -> X_t)]
iii) along the prior probability of being in state X_t-1,j [P(X_t-1,j)]

So the probability of being each word j for each t is the maximum probability from all the other words j at the previous order t-1: 
                        P(E_t, j) = max( P(X_t-1, :) * T(X_t-1 -> X_t, :) * EE(X_t -> E_t) )

for the first state transition, when there is no prior probability P(t-1, j), this is just set to 1's.
