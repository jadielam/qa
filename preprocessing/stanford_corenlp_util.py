"""Provides a way to tokenize text with Stanford CoreNLP.
"""

import json
import os
import preprocessing.constants as constants
import random
import re
import subprocess

from preprocessing.tokenized_word import *
from pycorenlp import StanfordCoreNLP
from util.string_util import utf8_str

_SHUFFLES_PER_PASSAGE = 3

# Debug function to print sentences in a list.
def _get_sentences(annotation_sentences):
    all_words = []
    for sentence in annotation_sentences:
        for token in sentence["tokens"]:
            all_words.append(token["word"])
    return " ".join([utf8_str(w) for w in all_words])


class StanfordCoreNlpCommunication():
    def __init__(self, data_dir):
        self.server_process = None
        self.data_dir = data_dir
        self.nlp = None

    def start_server(self):
        command = [ "java", "-cp",
        os.path.join(self.data_dir,
                "stanford-corenlp-full-2017-06-09/*"),
        "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", constants.STANFORD_CORENLP_PORT,
        "-quiet"]
        self.server_process = subprocess.Popen(command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        if self.server_process.poll() is not None:
            raise Exception("Couldn't start Stanford CoreNLP server.")
        print("Started Stanford CoreNLP server on port " + constants.STANFORD_CORENLP_PORT)
        self.nlp = StanfordCoreNLP('http://localhost:' + constants.STANFORD_CORENLP_PORT)

    def stop_server(self):
        print("Killed Stanford CoreNLP server")
        self.server_process.kill()

    def tokenize_list(self, text_list):
        output_list = []
        i = 0
        for text in text_list:
            output_list.append(self.tokenize_text(text))
            i += 1
            print("Progress", i, "out of", len(text_list))
        return output_list

    def _get_tokenized_words(self, annotation, shuffle):
        sentences = []
        num_output_sentences = 1 if not shuffle else _SHUFFLES_PER_PASSAGE
        for z in range(num_output_sentences):
            tokens = []
            s = annotation["sentences"]
            if shuffle and z > 0: # Keep the first example unshuffled.
                random.shuffle(s)
            for sentence in s:
                for token in sentence["tokens"]:
                    tokens.append(TokenizedWord(
                        token["word"],
                        token["characterOffsetBegin"],
                        token["characterOffsetEnd"],
                        token["ner"],
                        token["pos"]))
            sentences.append(tokens)
        return sentences

    def _perform_tokenize_text(self, text, shuffle):
        annotate = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,pos,ner',
            'outputFormat': 'json',
            'tokenize.language': 'English',
            'splitHyphenated': True,
            'tokenize.options': 'splitHyphenated=true,untokenizable=allKeep,invertible=true',
        })
        if isinstance(annotate, str):
            print("Some internal failure happened when using Stanford CoreNLP")
            return None
        return self._get_tokenized_words(annotate, shuffle)

    def tokenize_text_into_shuffled_sentences(self, text):
        """Input: A string

           Output: A list of lists of TokenizedWord's from the text that are
           shuffled sentences of the text.
        """
        return self._perform_tokenize_text(text, shuffle=True)

    def tokenize_text(self, text):
        """Input: A string

           Output: A list of TokenizedWord's from the text.
        """
        return self._perform_tokenize_text(text, shuffle=False)
