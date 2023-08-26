AADict={'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

def DPC(seq):
    dpc = list[0 for i in range(400)]
    l = len(seq)
    for i in range(l - 1):
        dpc[AADict[seq[i]] * 20 + AADict[seq[i + 1]]] += 1
    return dpc

fasta_path = 'data/final.fasta'
encoded = {}
for record in SeqIO.parse(fasta_path, 'fasta'):
    encoded[record.seq] = DPC(record.seq)
