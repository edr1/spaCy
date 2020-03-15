#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function
import plac
import spacy
from spacy import displacy
from spacy.tokens import Span
import random
from spacy.util import minibatch, compounding
from pathlib import Path
import os
from FTD_data_test import TRAIN_DATA_ALL


@plac.annotations(
    model=("Model name. Defaults to pt model.", "option", "m", str),
)
def main(model=None):
    print("--------------- PTBT ------------------------")
    if model is None:
        #model = "pt_core_news_sm"
	model = "/home/centos/spaCy/ftd/imex-ner-2"

    print("testing with ",model)
    nlp = spacy.load(model)

    for text, _ in TRAIN_DATA_ALL:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    #for token in doc:
    #    print(token.text.ljust(15,' ')+"\t"+token.lemma_+"\t"+token.pos_+"\t"+(token.tag_+"["+str(spacy.explain(token.tag_))+"]").ljust(50,' ')+"\t\t"+token.dep_.ljust(20,' ')+"\t"+token.shape_+"\t"+str(token.is_alpha)+"\t"+str(token.is_stop))
    #print("---------------------------------------")
    #for ent in doc.ents:
    #    print(ent.text.ljust(15,' ')+"\t"+(ent.label_+"["+str(spacy.explain(ent.label_))+"]").ljust(50,' '))

if __name__ == "__main__":
    plac.call(main)
