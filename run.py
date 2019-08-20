#import functionality

import torch
import torch.optim as optim
import torch.nn as nn

import os
import random 
import pickle

import models
from train import *

# setup parameters
SEED = 1234
#DATA_DIR = 'data_python'
#DATASET = 'sub_folder'f
DATA_DIR = '../AST-parsing/Data'
#DATASET = 'data_small'
#DATASET = 'data_numpy_width20_20'
#DATASET = 'data_numpy_pandas_sklearn_width_100_20'
FOLDER = 'split'
IR_NAME = 'IR_data_numpy_pandas_sklearn_width_100_20_wVal'
EMBEDDING_DIM = 128
DROPOUT = 0
#DROPOUT = 0.0
BATCH_SIZE = 500
CHUNKS = 1
MAX_LENGTH = 100
LOG_EVERY = 100 #print log of results after every LOG_EVERY batches
N_EPOCHS = 500
LOG_DIR = 'logs'
SAVE_DIR = 'checkpoints'
LOG_PATH = os.path.join(LOG_DIR, f'{DATASET}-{FOLDER}-log.txt')
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{DATASET}-{FOLDER}-model.pt')
LOAD = False # set true if you want to load model from MODEL_SAVE_PATH

# set random seeds for reproducability
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#load counts of each token in dataset

# with open(f'{DATA_DIR}/{DATASET}/dictionaries_all', 'rb') as file:
#     path_dictionary = pickle.load(file)
#     word2count = pickle.load(file)
#     path2count = pickle.load(file)
#     target2count = pickle.load(file)
#     n_training_examples = pickle.load(file)
with open(f'{DATA_DIR}/{DATASET}/{FOLDER}/dictionaries_train', 'rb') as file:
    n_training_examples = pickle.load(file)
    word2count = pickle.load(file)
    path2count = pickle.load(file)
    target2count = pickle.load(file)

print(f'Total training examples: {n_training_examples}')

# create vocabularies, initialized with unk and pad tokens
word2idx = {'<unk>': 0, '<pad>': 1}
path2idx = {'<unk>': 0, '<pad>': 1 }
target2idx = {'<unk>': 0, '<pad>': 1}

idx2word = {}
idx2path = {}
idx2target = {}

for w in word2count.keys():
    word2idx[w] = len(word2idx)

for k, v in word2idx.items():
    idx2word[v] = k

for p in path2count.keys():
    path2idx[p] = len(path2idx)

for k, v in path2idx.items():
    idx2path[v] = k

for t in target2count.keys():
    target2idx[t] = len(target2idx)

for k, v in target2idx.items():
    idx2target[v] = k

#device = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = models.Code2VecIgnoreVal(len(word2idx), len(path2idx), EMBEDDING_DIM, len(target2idx), DROPOUT)
model = models.Code2Vec(len(word2idx), len(path2idx), EMBEDDING_DIM, len(target2idx), DROPOUT)

if LOAD:
    print(f'Loading model from {MODEL_SAVE_PATH}')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

if not os.path.isdir(f'{LOG_DIR}'):
    os.makedirs(f'{LOG_DIR}')

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

for epoch in range(N_EPOCHS):

    log = f'Epoch: {epoch+1:02} - Training'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

    train_loss, train_acc, train_p, train_r, train_f1 = train(model, f'{DATA_DIR}/{DATASET}/{FOLDER}/train', optimizer,
                                                              criterion, BATCH_SIZE, CHUNKS, LOG_EVERY, LOG_PATH,
                                                              MAX_LENGTH, idx2target, word2idx, path2idx, target2idx,
                                                              n_training_examples, device)

    log = f'Epoch: {epoch+1:02} - Validation'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, f'{DATA_DIR}/{DATASET}/{FOLDER}/validation',
                                                                 criterion, BATCH_SIZE, CHUNKS, LOG_EVERY, LOG_PATH,
                                                                 MAX_LENGTH, idx2target, word2idx, path2idx, target2idx, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    log = f'| Epoch: {epoch+1:02} |\n'
    log += f'| Train Loss: {train_loss:.3f} | Train Precision: {train_p:.3f} | Train Recall: {train_r:.3f} | Train F1: {train_f1:.3f} | Train Acc: {train_acc*100:.2f}% |\n'
    log += f'| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p:.3f} | Val. Recall: {valid_r:.3f} | Val. F1: {valid_f1:.3f} | Val. Acc: {valid_acc*100:.2f}% |'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

log = 'Testing'
with open(LOG_PATH, 'a+') as f:
    f.write(log+'\n')
print(log)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc, test_p, test_r, test_f1 = evaluate(model, f'{DATA_DIR}/{DATASET}/{FOLDER}/test', criterion,
                                                        BATCH_SIZE, CHUNKS, LOG_EVERY, LOG_PATH, MAX_LENGTH,
                                                        idx2target, word2idx, path2idx, target2idx, device)

log = f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p:.3f} | Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f} | Test Acc: {test_acc*100:.2f}% |'
with open(LOG_PATH, 'a+') as f:
    f.write(log+'\n')
print(log)


"""
Get intermediate representations
"""
# paras = list(model.parameters())

examples = []

for example_name, example_body, example_length in file_iterator(f'{DATA_DIR}/{DATASET}/{FOLDER}/train', MAX_LENGTH):

    examples.append((example_name, example_body, example_length))

tensor_n, tensor_l, tensor_p, tensor_r, mask = numericalize_mm(examples, MAX_LENGTH, word2idx, path2idx, target2idx)

# place on gpu
tensor_n = tensor_n.to(device)
tensor_l = tensor_l.to(device)
tensor_p = tensor_p.to(device)
tensor_r = tensor_r.to(device)

# get code vector
temp = model.get_code_vec(tensor_l, tensor_p, tensor_r)
vectors = temp.detach().numpy()
print(vectors)
print(vectors.shape)

# save into file
with open(IR_NAME, 'wb') as file:
    pickle.dump(vectors, file)


