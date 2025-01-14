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

(A) Sentence Generation: 
The sentence is generated using a Markov model of the Bayesian Network by sampling from the probability P(x_t = k | x_t-1 = j, x_t-2 = i)

For the first word, the sentence start will always be the character "<s>", treated as a word in the vocab.txt
The words will be generated the prior sample: 
P(x_1...x_n) = product_i=1 to n ( P (x_i | parents(x_i) ) 

(B) Sentence Correction
