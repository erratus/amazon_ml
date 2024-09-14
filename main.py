import pandas as pd
import spacy
from spacy.tokens import DocBin
df=pd.read_csv("train_split.csv")
nlp = spacy.blank("en")
db = DocBin()

for index, row in df.iterrows():
    text = row['Text']
    entity_name = row['Entity Name']
    entity_value = row['Entity Value']

    start_idx = text.find(entity_value)
    end_idx = start_idx + len(entity_value)

    if start_idx != -1:
        doc = nlp.make_doc(text)
        ents = [(start_idx, end_idx, entity_name)]

        entities = []
        for start, end, label in ents:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                entities.append(span)

        doc.ents = entities
        db.add(doc)

db.to_disk("./train.spacy")