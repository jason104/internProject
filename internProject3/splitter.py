from sklearn.model_selection import train_test_split
import numpy as np
import json
import pandas as pd

class_tag = {'3.1.3': 0, '3.1.4': 1, '3.1.8': 2, '3.6.1': 3, '3.9.1': 4}

#def read_fasta_to_kmers(fasta_path, k=3, overlap=True):
#    r = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))
#    return r
#record = read_fasta_to_kmers('./p4.fasta')

train_dict, test_dict = {}, {}
data = pd.read_csv('./swissTremblPro.tsv', sep='\t')
for i, row in data.iterrows():
    if row.loc['Reviewed'] == 'reviewed':
        ecList = row.loc['EC number'].split('; ')
        for num in ecList:
            if num[:5] in class_tag:
                test_dict[row.loc['Sequence']] = class_tag[num[:5]]
                break     
    elif int(row.loc['Length']) < 800:
        ecList = row.loc['EC number'].split('; ')
        for num in ecList:
            if num[:5] in class_tag:
                train_dict[row.loc['Sequence']] = class_tag[num[:5]]

#for record in SeqIO.parse('./p4.fasta', 'fasta'):
#    if str(record.seq) not in x:
#        y.append(class_tag[str(record.id).split('_')[0]])
#        x.append(str(record.seq))
#for k, v in record.items():
#    x.append(str(v.seq))
#    y.append(class_tag [k.split('_')[0]])
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

#train_dict = dict(zip(x_train, y_train))
#test_dict = dict(zip(x_test, y_test))
with open('trainingPro.txt', 'w') as file1:
    file1.write(json.dumps(train_dict))
with open('testingPro.txt', 'w') as file2:
    file2.write(json.dumps(test_dict))
