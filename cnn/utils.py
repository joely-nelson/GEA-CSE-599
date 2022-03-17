import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import sys
import pandas as pd
import re
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import time
from nltk import ngrams
import itertools

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# MISC. FUNCTIONS
# ------------------------------------------------------------------------------
def time_str(secs):
    '''
    Given a time in seconds, prints the format in xh ym zs
    '''
    p_secs = int(secs % 60)
    p_mins = int((secs // 60) % 60)
    p_hours = int((secs // 60 // 60))
    
    return str(p_hours) + "h " + str(p_mins) + "m " + str(p_secs) + "s"
    
# ------------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------------------------------------------------------
def load_data_train():
    '''
    Load the data inside of the data folder for training and validation
    RETURNS:
        A tuple with 2 sequences
        - seqs: a list of the sequences
        - labels: a list of the labels
        NOTE: labels[i] will correspond to seqs[i]
    '''
    # import data
    df_train_values = pd.read_csv('../data/train_values.csv')
    df_train_labels = pd.read_csv('../data/train_labels.csv')

    # merge just to make sure that every label corresponds to the correct
    # sequence
    df_train = df_train_values.merge(df_train_labels,
                                    left_on='sequence_id', 
                                    right_on='sequence_id', 
                                    how = 'outer')

    # undo the one hot encoding in df_train_labels
    col_list = list(df_train.columns)
    label_cols = col_list[col_list.index('00Q4V31T'):]
    labels = df_train[label_cols].idxmax(1)

    seqs = list(df_train_values['sequence'])
    labels = list(labels)

    return seqs, labels

# ------------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS 
# ------------------------------------------------------------------------------
def label_to_count(labels):
    '''
    Given a list of labels, returns a dictionary that maps each class label 
    to how many instances of that label were present in the list.
    '''
    label_to_count_dict = {}
    for label in labels:
        if label not in label_to_count_dict:
            label_to_count_dict[label] = 0
        label_to_count_dict[label] += 1
    return label_to_count_dict


def prepare_kmers(seqs, k):
    '''
    Given a list of sequences, will turn them into a list of kmers instead
    ARGS:
        - seqs: a list of strings where every string is a sequence
        - k: the size of kmer
    RETURNS:
        - for each sequence, a list of tuples of the kmers
    EXAMPLE:
        >>> prepare_kmers(['ATCG', 'CCCC'], 2)
        [
         [('A', 'T'), ('T', 'C'), ('C', 'G')],
         [('C', 'C'), ('C', 'C'), ('C', 'C')], 
        ]
    '''
    return [list(ngrams(seq, k)) for seq in seqs]
    

def prepare_data(seqs):
    '''
    Given a list of sequences, will turn into a tokenized vector.
    
    ARGS:
        - seqs: a list of strings where every string is a sequence or token
    RETURNS:
        - tokenized_seqs (list(list(int))): list of list of tokens
        - voc2ind (dict) a dictionary where keys are letters, values are the 
          corresponding token
    '''
    max_len = 0
    
    # build up a voc2ind (letters:token)
    # based on ATGC and include padding and unknown tokens
    # voc2ind = {voc:ind for ind, voc in 
    #            enumerate(['<pad>', '<unk>', 'A', 'T', 'C', 'G'])}
    voc2ind = {'<pad>':0}

    # for printing
    total_seqs = len(seqs)
    curr_progress = 0
    curr_seq = 0
    
    i = len(voc2ind)
    
    # tokenize the sequences
    tokenized_seqs = []
    for seq in seqs:
        tokenized_seq = []
        for e in seq:
            # make sure the sequence is upper case, a == A
            # seq = seq.upper()
            # if we haven't seen this letter before, add to the corupus
            if not e in voc2ind:
                voc2ind[e] = i
                i += 1
            tokenized_seq.append(voc2ind[e])
        tokenized_seqs.append(tokenized_seq)
        
        # print progress
        curr_seq += 1
        new_progress = round(curr_seq / total_seqs, 2)
        if new_progress != curr_progress:
          curr_progress = new_progress
          # print(curr_progress)
        
    return tokenized_seqs, voc2ind


def prepare_labels(labels):
    '''
    Given a list of labels will turn them into integer labels
    Args:
        - labels: a list of labels
    Returns:
        - tokenized_labels: numpy array(list) a list of label tokens
        - label2token: (dict) a dictionary where keys are letters, values are 
          corresponding token
    '''
    print("Begin prepare labels")
    # for printing
    total = len(labels)
    curr_progress = 0
    curr = 0

    tokenized_labels = []
    label2token = {}
    i = 0
    for label in labels:
        if not label in label2token:
            label2token[label] = i
            i += 1
        tokenized_labels.append(label2token[label])


    return tokenized_labels, label2token


def pad(tokenized_seqs, voc2ind):
    '''
    Pad each sequence to the maximum length by adding a <pad> token
    
    ARGS:
        - tokenized_seqs (list(list(str))): list of list of tokens
        - voc2ind (dict) a dictionary where keys are letters, values are the 
          corresponding token
    RETURNS:
        a numpy array of all the tokenized sequences that have been padded to be 
        the same length.
    '''
    padded_seqs = []
    
    # find max sequence length
    print("Finding Max Sequence Length")
    max_len = 0
    min_len = math.inf
    for seq in tokenized_seqs:
        max_len = max(len(seq), max_len)
        min_len = min(len(seq), min_len)
    print("Max sequence length found:", max_len)
    print("Min sequence length found:", min_len)
    
    # add padding so sequences are max_length
    print("Begin Padding")
    for seq in tokenized_seqs:
        padded_seq = seq + [voc2ind['<pad>']] * (max_len - len(seq))
        padded_seqs.append(padded_seq)
    
    print("Change to a numpy array")
    return np.array(padded_seqs, dtype=np.float32)


def data_loader(train_inputs, val_inputs, train_labels, val_labels,
                batch_size=50):
    """
    Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, val_inputs, train_labels, val_labels])

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def prep_data(seqs, labels, cutoff_len=None, kmer_size=None):
    '''
    prepares data and returns train-dataloader and test_dataloader
    given the cutoff_len. 
    If no cutoff_len is given, will not cut sequences.
    If no kmer_size is given, will not convert to kmers. Otherwise
    will conver to kmers of that length.
    '''
    
    # cutting off sequences to given length
    if not cutoff_len is None:
        seqs = [seq[:cutoff_len] for seq in seqs]  
        
    # prepare kmers
    if not kmer_size is None:
        seqs = prepare_kmers(seqs, kmer_size)
        
    # tokenizing and getting a vocab
    tokenized_seqs, voc2ind = prepare_data(seqs)
    print("data prepared")
    
    # if we are preparing kmers, we need to make sure we include
    # kmers that are not included in the input sequences so that if we
    # read in sequences with new k-mers
    # basically will add new kmers to voc2ind (dict) a 
    # dictionary where keys are letters, values are the corresponding token
    if not kmer_size is None:
        # get all kmers
        bases = ['A', 'C', 'G', 'T']
        all_kmers = itertools.combinations(bases, kmer_size)
        # add them into voc2ind if not present
        i = len(voc2ind)
        for kmer in all_kmers:
            if kmer not in voc2ind:
                voc2ind[kmer] = i
                i += 1
                
    # padding
    tokenized_seqs = pad(tokenized_seqs, voc2ind)
    print("tokenized sequenced")

    # tokenizing labels
    tokenized_labels, label2token = prepare_labels(labels)
    print('tokenized labels')

    print()

    # Showing the result of this:
    print("\n", tokenized_seqs, 
          "\n\n", voc2ind, 
          "\n\n", label_to_count(labels))

    # Train Test Split
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        tokenized_seqs, tokenized_labels, test_size=0.1, random_state=42)

    # Load data to PyTorch DataLoader
    train_dataloader, test_dataloader = data_loader(train_inputs, test_inputs, 
                                                    train_labels, test_labels, 
                                                    batch_size=50)
    
    return train_dataloader, test_dataloader, seqs, voc2ind, tokenized_seqs, labels

# ------------------------------------------------------------------------------
# TRAINING FUNCTIONS 
# ------------------------------------------------------------------------------

def train(net, dataloader, device, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
  ''' Trains a neural network. Returns a 2d numpy array, where every list 
  represents the losses per epoch.
  '''
  net.to(device)
  losses_per_epoch = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  for epoch in range(epochs):
    sum_loss = 0.0
    losses = []
    for i, batch in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())
        sum_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            if verbose:
              print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
    # print(len(losses))
    losses_per_epoch.append(np.mean(losses))
  return losses_per_epoch


def accuracy(net, dataloader):
    '''
    Given a trained neural network and a dataloader, computes the accuracy.
    Arguments:
        - net: a neural network
        - dataloader: a dataloader
    Returns:
        - fraction of examples classified correctly (float)
        - number of correct examples (int)
        - number of total examples (float)
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total, correct, total


def print_eval(net, train_dataloader, test_dataloader):
    '''
    Given a test and train data loader, prints the test and train accuracy and
    the number of examples they got right.
    RETURNS
        (train_acc, test_acc) results of running accuracy on the two dataloaders
    '''
    train_acc = accuracy(net, train_dataloader)
    test_acc = accuracy(net, test_dataloader)
    

    print("Train accuracy: " + str(train_acc[0]) + "\t(" + str(train_acc[1]) + "/" + str(train_acc[2]) + ")")
    print("Test accuracy: " + str(test_acc[0]) + "\t(" + str(test_acc[1]) + "/" + str(test_acc[2]) + ")")
          
    return train_acc, test_acc


def plot_losses(losses, smooth_val = None, title = ""):
    '''
    Plots the losses per epoch returned by the training function.
    Args:
        losses: a list of losses returned by train
        smooth_val: an optinal integer value if smoothing is desired
        title: a title for the graph
    '''
    # loss = np.mean(losses, axis = 1)
    epochs = [i for i in range(1, len(losses) + 1)]
    if smooth_val is not None:
        lossses = smooth(losses, smooth_val)
    plt.plot(epochs, losses, marker="o", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    

def smooth(x, size):
    '''
    Given an array, smooths it by some number size, to make it look less janky.
    '''
    return np.convolve(x, np.ones(size)/size, mode='same')
