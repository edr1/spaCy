#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function
import spacy
from spacy import displacy
from spacy.tokens import Span
import random
from spacy.util import minibatch, compounding
from pathlib import Path
import os

def testENUS():
    print("--------------- ENUS ------------------------")
    nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load("/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_2")
    #doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    #doc = nlp("Google and fb are hiring a new vice president of global policy")
    #doc = nlp("I like London and Berlin")
    #doc = nlp("let me google this for you")

    #doc = nlp("Google Maps launches location sharing where you can find Horses")
    #doc = nlp("Uber blew through $1 million a week")ll
    doc = nlp("Android Pay expands to Canada")
    #doc = nlp("Who is Shaka Khan from Uber")
    #doc = nlp("Spotify steps up Asia expansion")
    #doc = nlp("Horses are too tall and they pretend to care about your feelings")
    #doc = nlp("horses pretend to care about your feelings")
    for token in doc:
        print(token.text.ljust(15,' ')+"\t"+token.lemma_+"\t"+token.pos_+"\t"+(token.tag_+"["+str(spacy.explain(token.tag_))+"]").ljust(50,' ')+"\t\t"+token.dep_.ljust(20,' ')+"\t"+token.shape_+"\t"+str(token.is_alpha)+"\t"+str(token.is_stop))
    print("---------------------------------------")
    #fb_ent = Span(doc, 2, 3, label="ORG")
    #doc.ents = list(doc.ents) + [fb_ent]
    for ent in doc.ents:
        print(ent.text.ljust(15,' ')+"\t"+(ent.label_+"["+str(spacy.explain(ent.label_))+"]").ljust(50,' '))
    #displacy.serve(doc, style="ent")

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
