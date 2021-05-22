"""
NLTK: Parts of speech tagging
    
Summary:
    - POS Tagging in NLTK is a process to mark up the words in text format for a 
    particular part of a speech based on its definition and context.
    - Some NLTK POS tagging examples are: CC, CD, EX, JJ, MD, NNP, PDT, PRP$, TO, etc.
    - POS tagger is used to assign grammatical information of each word of the 
    sentence. Installing, Importing and downloading all the packages of Part of Speech tagging with NLTK is complete.
"""

import nltk

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

words = nltk.word_tokenize(paragraph)

# Generate POS tagging 
# Example: [(WORD, TAGGING), (WORD, TAGGING),...]
pos_words = nltk.pos_tag(words)

pos_paragraph = " ".join([f"{word[0]}_{word[1]}" for word in pos_words])

"""
Here are the meanings of the Parts-Of-Speech tags used in NLTK

CC: Coordinating conjunction
CD: Cardinal number
DT: Determiner
EX: Existential there
FW: Foreign word
IN: Preposition or subordinating conjunction
JJ: Adjective
JJR: Adjective, comparative
JJS: Adjective, superlative
LS: List item marker
MD: Modal
NN: Noun, singular or mass
NNS: Noun, plural
NNP: Proper noun, singular
NNPS: Proper noun, plural
PDT: Predeterminer
POS: Possessive ending
PRP: Personal pronoun
PRP$: Possessive pronoun
RB: Adverb
RBR: Adverb, comparative
RBS: Adverb, superlative
RP: Particle
SYM: Symbol
TO: to
UH: Interjection
VB: Verb, base form
VBD: Verb, past tense
VBG: Verb, gerund or present participle
VBN: Verb, past participle
VBP: Verb, non-3rd person singular present
VBZ: Verb, 3rd person singular present
WDT: Wh-determiner
WP: Wh-pronoun
WP$: Possessive wh-pronoun
WRB: Wh-adverb
"""
