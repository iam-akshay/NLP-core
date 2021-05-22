"""
----------------------------------------------
Natural Language Processing using NLTK python
----------------------------------------------

Install nltk: pip install nltk

NLTK (Natural Language Toolkit) is a suite that contains libraries and programs for statistical language processing. It is one of the most powerful NLP libraries, which contains packages to make machines understand human language and reply to it with an appropriate response.

Content:
    - sent_tokenize
    - word_tokenize
    - Stemming
    - Lematization
    - Stop Words
"""
import nltk

#The downloader will search for an existing nltk_data directory to install NLTK data.
nltk.download()

paragraph = """Thank you all so very much. Thank you to the Academy. Thank you to all of you in this room. I have to congratulate the other incredible nominees this year for their unbelievable performances. The Revenant was the product of the tireless efforts of an unbelievable cast and crew I got to work alongside. First off, to my brother in this endeavor, Mr. Tom Hardy. Tom, your fierce talent on screen can only be surpassed by your friendship off screen. To Mr. Alejandro Innaritu, as the history of cinema unfolds, you have forged your way into history these past 2 years… thank you for creating a transcendent cinematic experience. Thank you to everybody at Fox and New Regency…my entire team. I have to thank everyone from the very onset of my career…to Mr. Jones for casting me in my first film to Mr. Scorsese for teaching me so much about the cinematic art form. To my parents, none of this would be possible without you. And to my friends, I love you dearly, you know who you are. Making The Revenant was about man’s relationship to the natural world. A world that we collectively felt in 2015 as the hottest year in recorded history. Our production needed to move to the southern tip of this planet just to be able to find snow. Climate change is real, it is happening right now. It is the most urgent threat facing our entire species, and we need to work collectively together and stop procrastinating. We need to support leaders around the world who do not speak for the big polluters, but who speak for all of humanity, for the indigenous people of the world, for the billions and billions of underprivileged people out there who would be most affected by this. For our children’s children, and for those people out there whose voices have been drowned out by the politics of greed. I thank you all for this amazing award tonight. Let us not take this planet for granted. I do not take tonight for granted. Thank you so very much."""

# return sentences from whole paragraph
# split based on space " "
sentences = nltk.sent_tokenize(paragraph)

# return all the words from the paragraph
#split based on dot "." and comma ","
words = nltk.word_tokenize(paragraph)


"""
Stemming: It is the process of reducing infected or derived words to their
word steam, base or root form. 

Example: 
    - Intelligence, Intelligent, Intelligently -> Intelligen
    - going, goes, gone -> go
    - final, finally -> fina
    
Here we can say that after word stemming, stem word doesn't have any meaning
like Inteligen, fina. So here we can use Lemmatization for that.

- It is use, when word meaning is not important while analysis,
example: Spam detection
- Take less time
"""

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
sentences = nltk.sent_tokenize(paragraph)
for index, sentence in enumerate(sentences):
    words = nltk.word_tokenize(sentence)
    words = [stemmer.stem(word) for word in words]
    sentences[index] = " ".join(words)


"""
Lematization: It is same as stemming, but intermidiate representation/root
form has a meaning.

Example: 
    - Intelligence, Intelligent, Intelligently -> Inteligent
    
- Slow compare to stemming
- Used when word meaning is imp. in analysis. Example: Q&A application
"""
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
for index, sentence in enumerate(sentences):
    words = nltk.word_tokenize(sentence)
    words = [lemma.lemmatize(word) for word in words]
    sentences[index] = " ".join(words)
    
    
"""
Stop words removal: Words that is not having any specific meaning, and it comes
many times in the sentences like to, is, be, the, etc.
"""
    
from nltk.corpus import stopwords
sentences = nltk.sent_tokenize(paragraph)

for index, sentence in enumerate(sentences):
    words = nltk.word_tokenize(sentence)
    new_words = [word for word in words if word not in stopwords.words('english')]
    sentences[index] = " ".join(new_words)
