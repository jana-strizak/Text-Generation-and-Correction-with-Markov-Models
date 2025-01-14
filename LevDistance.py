# implementing the Levenshtein Distance 
import numpy as np 


def levDist(word1, word2):
    lenW1 = len(word1)
    lenW2 = len(word2)

    d = np.zeros([lenW1 + 1, lenW2 + 1])

    for i in range(0, lenW1+1):
        d[i, 0] = i

    for j in range(0, lenW2+1):
        d[0, j] = j

    for i in range(1, lenW1+1):
        for j in range(1, lenW2+1):
            
            # check if elements are the same 
            if word1[i-1] == word2[j-1]:
                cost = 0
            else:
                cost = 1
            
            # update element in d matrix
            d[i,j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)

    #print(d)
    levDist = d[lenW1, lenW2]
    return(levDist)


#word1 = "sunday"
#word2 = "saturday"

word1 = "kitten"
word2 = "sitting"

print(levDist(word1, word2))