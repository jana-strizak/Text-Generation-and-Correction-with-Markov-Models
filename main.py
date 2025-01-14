#%%
from functions import sentenceMaker, NGramModel

# define model for probabilistic sentence editing 
# note this distribution is heavily from Jane Austen's Emma, so generated sentence are that of the book
vocabFile = "data/vocab.txt"
unigramFile = "data/unigram_counts.txt"
bigramFile = "data/bigram_counts.txt"
trigramFile = "data/trigram_counts.txt"
model = NGramModel(vocabFile, unigramFile, bigramFile, trigramFile)
sentence = sentenceMaker(model)

# %%
# PART A: Sentence Generation
sentenceGenerated, distUse = sentence.generate_sentence(show=0)

# display
print(" The generated Sentence is: ")
print(sentenceGenerated)
print("Each word was sampled from the following distribution: ")
print("".join(distUse))

# %%
# PART B: Sentence Correction 
sentenceWrongs = ["I think hat twelve thousand pounds", "she haf heard them", "She was ulreedy quit live", "John Knightly wasn’t hard at work", "he said nit word by"]
for s in sentenceWrongs:
    print("Input Sentence is: " + s)
    sentenceFixed = sentence.fixSentence(s)
    print("Corrected Sentence is:" + sentenceFixed)
    print("")
# %%
