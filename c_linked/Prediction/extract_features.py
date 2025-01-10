from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import os
import csv

def getProtT5(pdb_id, seq):
  device = torch.device('cpu')

  # Load the tokenizer
  tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

  # Load the model
  model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

  chunks = [seq[i:i+8797] for i in range(0,len(seq), 8797)]
  final = np.zeros((1,1024))
  for chunk in chunks:
    sequence_examples = [chunk]

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
      embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)


    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0,:len(chunk)] # shape (7 x 1024)

    print(emb_0.size())

    # Assuming last_hidden_state is your tensor
    last_hidden_state_np = emb_0.cpu().detach().numpy()
    final = np.concatenate((final,last_hidden_state_np), axis=0)

  final = np.delete(final,0,axis = 0)
  return final


def getESM2(pdb_id, seq):

  model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

  batch_converter = alphabet.get_batch_converter()
  model.eval()  # disables dropout for deterministic results

  chunks = [seq[i:i+1024] for i in range(0,len(seq), 1024)]
  final = np.zeros((1,1280))
  for chunk in chunks:
      data = []
      data.append(("protein1", chunk))

      batch_labels, batch_strs, batch_tokens = batch_converter(data)
      batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

      # Extract per-residue representations (on CPU)
      with torch.no_grad():
          results = model(batch_tokens, repr_layers=[33], return_contacts=True)
      token_representations = results["representations"][33]

      rep = token_representations.cpu().detach().numpy()[0][1:-1]
      final = np.concatenate((final,rep), axis=0)

  final = np.delete(final,0,axis = 0)
  print(final.shape)
  return final


with open('dataset.txt','r') as f, open('prott5.csv', 'w') as out:
  lines = f.readlines()

  for i in range(0, len(lines), 2):
    features = getProtT5(lines[i].rstrip().split(',')[0], lines[i+1].rstrip())
    splitter = lines[i].rstrip().split(',')
    print(features.shape)
    print(lines[i].rstrip().split(',')[0], i // 2 + 1)
    for j in range(1, len(splitter)):
      out.write(','.join([str(num) for num in features[int(splitter[j])]]) + '\n')


with open('dataset.txt','r') as f, open('esm2.csv', 'w') as out:
  lines = f.readlines()

  for i in range(0, len(lines), 2):
    features = getESM2(lines[i].rstrip().split(',')[0], lines[i+1].rstrip())
    splitter = lines[i].rstrip().split(',')
    print(features.shape)
    print(lines[i].rstrip().split(',')[0], i // 2 + 1)
    for j in range(1, len(splitter)):
      out.write(','.join([str(num) for num in features[int(splitter[j])]]) + '\n')

window_size = 7
mapper = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
    '-': 20 # For padding
}

with open('dataset.txt','r') as f, open('word.csv', 'w') as out:
  lines = f.readlines()

  for i in range(0, len(lines), 2):
    fasta = lines[i+1].strip()
    splitter = lines[i].rstrip().split(',')
    print(lines[i].rstrip().split(',')[0], i // 2 + 1)

    targets = []
    for j in range(1, len(splitter)):
        targets.append([splitter[j]])
    
    for t in targets:
      site_position = int(t[0]) + 1
      start = (site_position - window_size - 1, 0)[(site_position - window_size - 1) < 0]
      end = (site_position + window_size, len(fasta),)[site_position + window_size > len(fasta)]
      cutout = fasta[start:end]

      start_pad = site_position - window_size - 1
      end_pad = site_position + window_size - len(fasta)

      if start_pad < 0:
          cutout = '-' * abs(start_pad) + cutout
      if end_pad > 0:
          cutout = cutout + '-' * end_pad
      # print(cutout)
      
      embedding = []
      for aa in cutout:
          embedding += [mapper[aa]]
      embedding = np.array(embedding)
      # print(embedding.shape)
      out.write(','.join(map(str, embedding)) + '\n')

final1 = np.loadtxt('prott5.csv', delimiter=',', skiprows=0)
final2 = np.loadtxt('esm2.csv', delimiter=',', skiprows=0)
final3 = np.loadtxt('word.csv', delimiter=',', skiprows=0)
if final1.ndim == 1:
  final1 = np.array([final1])
if final2.ndim == 1:
  final2 = np.array([final2])
if final3.ndim == 1:
  final3 = np.array([final3])
final = np.concatenate((final1, final2, final3), axis=1)
np.savetxt('features.csv', final, delimiter=',')