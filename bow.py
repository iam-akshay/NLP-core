"""

BOW (Bag of Words) Model:
    BOW is a NLP technique of text modelling. Whenever we apply any algorithm in NLP, 
    it works on numbers. We cannot directly feed our text into that algorithm. 
    Hence, Bag of Words model is used to preprocess the text by converting it 
    into a bag of words, which keeps a count of the total occurrences of most 
    frequently used words.

Process to create BOW model:
    - Convert text in lower case.
    - Remove all non-word characters.
    - Remove all punctuations.
    - Now make table of each word with its appearance counts
    - Select number of most frequently used words.
    - Building the Bag of Words model
"""

import nltk
import re # Regular Expression
import heapq # Heap Queue
import numpy as np

paragraph = """Thank you all so very much. Thank you to the Academy. Thank you to all of you in this room. 
I have to congratulate the other incredible nominees this year for their unbelievable performances. The Revenant was the product of 
the tireless efforts of an unbelievable cast and crew I got to work alongside. First off, to my brother in this endeavor, 
Mr. Tom Hardy. Tom, your fierce talent on screen can only be surpassed by your friendship off screen. 
To Mr. Alejandro Innaritu, as the history of cinema unfolds, you have forged your way into history these past 
2 years… thank you for creating a transcendent cinematic experience. Thank you to everybody at Fox and New Regency…my 
entire team. I have to thank everyone from the very onset of my career…to Mr. Jones for casting me in my first film 
to Mr. Scorsese for teaching me so much about the cinematic art form. To my parents, none of this would be possible 
without you. And to my friends, I love you dearly, you know who you are. Making The Revenant was about man’s relationship
to the natural world. A world that we collectively felt in 2015 as the hottest year in recorded history. 
Our production needed to move to the southern tip of this planet just to be able to find snow. Climate change is 
real, it is happening right now. It is the most urgent threat facing our entire species, and we need to work 
collectively together and stop procrastinating. We need to support leaders around the world who do not speak 
for the big polluters, but who speak for all of humanity, for the indigenous people of the world, for the billions
and billions of underprivileged people out there who would be most affected by this. For our children’s children,
and for those people out there whose voices have been drowned out by the politics of greed. I thank you all for this
amazing award tonight. Let us not take this planet for granted. I do not take tonight for granted. Thank you so very
much."""

#-------------------------------
# Step1: Preprocess
#-------------------------------
dataset = nltk.sent_tokenize(paragraph)
for index, sentence in enumerate(dataset):
    # convert text to lower case and remove special characters
    sentence = re.sub('\W', ' ', sentence.lower())
    # remove extra spaces
    dataset[index] = re.sub("\s+", ' ', sentence)
    
#-------------------------------
# Step2: Creating the histogram
#-------------------------------
word_2_count = dict()
for sentence in dataset:
    words = nltk.word_tokenize(sentence)
    for word in words:
        word_2_count[word] = word_2_count[word] + 1 if word in word_2_count else 1
# N largest frequent words (for this making use of heapq)
frequent_words = heapq.nlargest(n=100, iterable=word_2_count, key=word_2_count.get)

#-------------------------------
# Step3: Create Bag of Words
#-------------------------------
bow_model = []
for sentence in dataset:
    vector = []
    for word in frequent_words:
        vector.append(1 if word in nltk.word_tokenize(sentence) else 0)
    bow_model.append(vector)
# create 2D array representation 
bow_model = np.asarray(bow_model)    
    