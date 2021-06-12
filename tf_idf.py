"""
TF-IDF: “Term Frequency — Inverse Document Frequency”.

This is a technique to quantify a word in documents, we generally compute a weight 
to each word which signifies the importance of the word in the document and corpus. 
This method is a widely used technique in Information Retrieval and Text Mining.

TF: 
    Formula: (No. of word in document/Total no. of word in document)

IDF:
    Formula: (log(Total no. of documents/No. of times word occurs))
    
Terminology:
-------------
    t — term (word)
    d — document (set of words)
    N — count of corpus
    corpus — the total document set
"""

import nltk
import re # Regular Expression
import heapq # Heap Queue
import numpy as np

paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""

corpus = nltk.sent_tokenize(paragraph)
for index, document in enumerate(corpus):
    document = re.sub('\W', ' ', document.lower())
    corpus[index] = re.sub('\s+', ' ', document)
    
word_2_count = {}
for document in corpus:
    for word in nltk.word_tokenize(document)    :
        word_2_count[word] = word_2_count[word]+1 if word in word_2_count else 1        
frequent_words = heapq.nlargest(n=100, iterable=word_2_count, key=word_2_count.get)

# IDF calculation
# idf = log(total no. of documents/no. of time word occurs) (for whole document)
word_idf = {}
for word in frequent_words:
    doc_count = 0
    for document in corpus:
        if word in nltk.word_tokenize(document):
            doc_count += 1
    no_of_document = len(corpus)
    word_idf[word] = np.log((no_of_document/doc_count) + 1)


# TF calculation
# tf = no. of time word occurs/total no. of words (for single docuement)
word_tf = {}
for sentence in corpus:
    word_with_count = {}
    for word in nltk.word_tokenize(sentence):
        if word in frequent_words:
            word_in_count[word] = word_in_count[word]+1 if word in word_in_count else 1
            
    