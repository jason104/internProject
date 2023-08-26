import numpy as np
from doc2vec import read_fasta_to_kmers, train_doc2vec


DS2_data = read_fasta_to_kmers('./data/final.fasta')

train_doc2vec(DS2_data, './model/Doc2Vec_model/AFP_doc2vec_DS2.model')
