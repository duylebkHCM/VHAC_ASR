import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\']'

def remove_special_characters(transcript):
    transcript = re.sub(chars_to_ignore_regex, '', transcript).lower()
    return transcript

import sys
import nltk
nltk.download('punkt')

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        jline = remove_special_characters(sentence)
        jline = ' '.join(nltk.word_tokenize(jline)).lower()
        print(jline)