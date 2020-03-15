#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import os

from FTD_data import TRAIN_DATA_ALL
from FTD_data2 import TRAIN_DATA_ALL_1M


@plac.annotations(
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def trainingNERModel (train_data=None, model=None, output_dir=None, n_iter=100):
    
    if model is None:
        model = "pt_core_news_sm";


    #Once you have a GPU-enabled installation, the best way to activate it is to call spacy.prefer_gpu or spacy.require_gpu() 
    #somewhere in your script before any models have been loaded. require_gpu will raise an error if no GPU is available.
    #spacy.prefer_gpu()
    print("Load the model", model, "set up the pipeline and train the entity recognizer.")
    nlp = spacy.load(model)

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly but only if we're
        # training a new model
        #if model is None:
        #    nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in train_data:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        #print("Loading from", output_dir)
        #nlp2 = spacy.load(output_dir)
        #for text, _ in train_data:
        #    doc = nlp2(text)
        #    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    current_path=os.getcwd()+"/imex-ner-2"
    print(current_path)
    trainingNERModel(TRAIN_DATA_ALL_1M, None, current_path, 100)

    #trainingNERModel(TRAIN_DATA_ALL, None, current_path, 100)
    #trainingNERModel(TRAIN_DATA_AUTOR_1E, None, current_path, 100)
    #trainingNERModel(TRAIN_DATA_DISCIPLINA_1E, current_path, current_path, 100)
    #trainingNERModel(TRAIN_DATA_EDITORA_1E, current_path, current_path, 100)
    #trainingNERModel(TRAIN_DATA_LIVRO_1E, current_path, current_path, 100)
