import spacy
from spacy import displacy
from spacy.tokens import Span
import random
from spacy.util import minibatch, compounding
from pathlib import Path
import plac
import os

from FTD_autores_dat import TRAIN_DATA_AUTOR_1E
from FTD_disciplinas_dat import TRAIN_DATA_DISCIPLINA_1E
from FTD_editoras_dat import TRAIN_DATA_EDITORA_1E
from FTD_livros_dat import TRAIN_DATA_LIVRO_1E


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
        # reset and initialize the weights randomly – but only if we're
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
    nlp = spacy.load("/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_pt_1")
    doc = nlp("SAT: DICIONARIO ESPANHOL FALSOS COGNATOS ESPANHOL")
    #doc = nlp("EDE: 1 LIVRO COL PARAISO DA CRIANCA LÍNGUA PORTUGUESA")
    #doc = nlp("LÍNGUA PORTUGUESA")
    #doc = nlp("SAT")
    #doc = nlp("1 LIVRO NAO ESPECIFICADO 6º ANO")
    #doc = nlp("João falou pra Maria")
    for token in doc:
        print(token.text.ljust(15,' ')+"\t"+token.lemma_+"\t"+token.pos_+"\t"+(token.tag_+"["+str(spacy.explain(token.tag_))+"]").ljust(50,' ')+"\t\t"+token.dep_.ljust(20,' ')+"\t"+token.shape_+"\t"+str(token.is_alpha)+"\t"+str(token.is_stop))
    print("---------------------------------------")
    for ent in doc.ents:
        print(ent.text.ljust(15,' ')+"\t"+(ent.label_+"["+str(spacy.explain(ent.label_))+"]").ljust(50,' '))


if __name__ == "__main__":
    #testENUS()
    #testPTBR()
    current_path=os.getcwd()+"/ftd/imex-ner-1"
    #trainingNERModel(TRAIN_DATA_AUTOR_1E, None, current_path, 100)
    trainingNERModel(TRAIN_DATA_DISCIPLINA_1E, None, current_path, 100)
    trainingNERModel(TRAIN_DATA_EDITORA_1E, current_path, current_path, 100)
    
