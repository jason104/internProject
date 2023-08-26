from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load('./model/Doc2Vec_model/AFP_doc2vec_DS2.model')
print(model.docvecs[100])
print(model.docvecs[1000])
