from __future__ import unicode_literals, print_function
import json
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pandas as pd

import csv
import ast


def train(model, file_name, output_dir ,n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # training data
    df = pd.read_csv(file_name, sep='\t', encoding = "unicode_escape")
    print(df)
    #with open(file_name, encoding="utf8") as df:
    TRAIN_DATA = []
    for index,rows in df.iterrows():
        TRAIN_DATA.append((rows['sentence'],ast.literal_eval(rows['entity'])))

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    
    #add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
           
            ner.add_label(ent[2]) 
            
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print('Losses', losses)
            
    #save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
