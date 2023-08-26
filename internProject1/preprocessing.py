import pandas as pd
from gensim.test.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


raw_data = pd.read_excel('nitrogen1.xlsx')

seqs = raw_data['Sequence'].values.tolist()
tagged_data = [TaggedDocument(words=[d], tags=[str(i)]) for i, d in enumerate(seqs)]
model = Doc2Vec(vector_size=300, epochs=50, min_count=1)
model.build_vocab(tagged_data)
# model.save('doc2vec.model')
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


print(model.infer_vector([seqs[100]]))

#print(model.docvecs[1])
#print(raw_data)

