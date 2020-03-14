# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import spacy
from spacy import displacy
from spacy.tokens import Span
import random
from spacy.util import minibatch, compounding
from pathlib import Path
import os


def testPTBR():
    print("--------------- PTBT ------------------------")
    #nlp = spacy.load("pt_core_news_sm")
    nlp = spacy.load("/home/centos/spaCy/ftd/imex-ner-1")
    doc = nlp("LAURE MONLOUBOU")
    for token in doc:
        print(token.text.ljust(15,' ')+"\t"+token.lemma_+"\t"+token.pos_+"\t"+(token.tag_+"["+str(spacy.explain(token.tag_))+"]").ljust(50,' ')+"\t\t"+token.dep_.ljust(20,' ')+"\t"+token.shape_+"\t"+str(token.is_alpha)+"\t"+str(token.is_stop))
    print("---------------------------------------")
    for ent in doc.ents:
        print(ent.text.ljust(15,' ')+"\t"+(ent.label_+"["+str(spacy.explain(ent.label_))+"]").ljust(50,' '))


if __name__ == "__main__":
    testPTBR()
