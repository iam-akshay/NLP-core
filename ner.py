"""
NLTK: 
    - NER (Named Entity Recognition)
    
Named entity recognition (NER)is probably the first step towards information 
extraction that seeks to locate and classify named entities in text into 
pre-defined categories such as the names of persons, organizations, locations, 
expressions of times, quantities, monetary values, percentages, etc.    

"""

import nltk

paragraph = "Hello my name is Akshay Jain and I was born in Rajasthan"
words = nltk.word_tokenize(paragraph)
pos_tagging = nltk.pos_tag(words)

# identify the Named Entity
ner_data = nltk.ne_chunk(pos_tagging)

# display in graphical view
ner_data.draw()

