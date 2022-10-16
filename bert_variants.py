#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Trains a BERT model variants using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.

To run this file,

1. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

python bert_variants.py

Running with the best hyperparams must give a macro F1-score of around 89/90.

Or if you want to change any of the arguments, please type

python bert_variants.py --help

If you want to try any other bert variant, please type

python bert_variants.py --langmodel_name distilbert-base-uncased


# microsoft/deberta-v3-base
# xlnet-base-cased
# roberta-base
# bert-base-uncased
# distilbert-base-uncased
# albert-base-v2
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

import random as python_random
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

import pandas as pd
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import seaborn as sn


def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    
    parser.add_argument("-t", "--test_file", type=str, default='test.txt',
                        help="If added, use trained model to predict on test set")
    
    parser.add_argument("-e", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Learning rate for the optimizer")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Number of epochs for training")

    parser.add_argument("--max_seq_len", default=200, type=int,
                        help="Maximum length of input sequence after BPE")

    parser.add_argument("--langmodel_name", default="roberta-base", type=str,
                        help="Name of the base pretrained language model")

    parser.add_argument("--trainable_bert_base", default=True, type=bool,
                        help="Set BERT weights to false and train only the classifier layer.\
                              Use this only for BERT. Otherwise, the script throws an error")

    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            if tokens == "": continue
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def get_lr_metric(optimizer):
    '''
    Function for printing LR after each epoch
    https://stackoverflow.com/questions/47490834/how-can-i-print-the-learning-rate-at-each-epoch-with-adam-optimizer-in-keras
    '''
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def create_model(args, Y_train, learning_rate = 5e-5, lm = "bert-base-uncased",
                 batchsize = 16, num_epochs = 2):
    """
    Create model and tokenizer using the hf model reponame
    Args:
        args: Command line argument passed
        Y_train: Training label data for creating the final classifier layer. 
                 Only length used
        learning_rate (float): Learning rate. Defaults to 5e-5.
        lm (str, optional): Language model name. Defaults to "bert-base-uncased".
        batchsize (int, optional): Batch size for batching input. Defaults to 16.
        num_epochs (int, optional): Number of epochs. Defaults to 2.

    Returns:
        model and its tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels= len(set(Y_train)))
    loss_function = CategoricalCrossentropy(from_logits=True)

    if not args.trainable_bert_base:
        print("Setting bert params to non trainable")
        model.bert.trainable = False

    starter_learning_rate = learning_rate
    end_learning_rate = learning_rate/100
    decay_steps = (len(Y_train)/batchsize)*num_epochs
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                                    starter_learning_rate, decay_steps,
                                    end_learning_rate, power=0.5)
    optim = Adam(learning_rate=learning_rate_fn)
    lr_metric = get_lr_metric(optim)
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy', lr_metric])
    return model, tokenizer


def train_model(model, tokens_train, Y_train, 
                tokens_dev, Y_dev, encoder, batchsize = 16, num_epochs = 2):
    """
    Train the model here
    And also validate for every epoch
    EarlyStopping is implemented here with patience of 3 epochs
    Returns:
        model: Trained model
    """
    verbose = 1
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Finally fit the model to our data
    model.fit(tokens_train, Y_train, verbose=1, epochs=num_epochs, callbacks=[earlystopping],
    batch_size=batchsize, validation_data=(tokens_dev, Y_dev))

    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, tokens_dev, Y_dev, "dev", encoder)
    return model


def calculate_confusion_matrix(Y_test, y_pred, labels):
    matrix = confusion_matrix(Y_test, y_pred)
    # Convert to pandas dataframe confusion matrix.
    matrix = (pd.DataFrame(matrix, index=labels, columns=labels))
    return matrix


def plot_confusion_matrix(matrix):
    fig, _ = plot.subplots(figsize=(9, 8))
    sn.heatmap(matrix, annot=True, cmap=plot.cm.Blues, fmt='g')
    # show the picture
    plot.show()
    fig.savefig("lstm-heatmap.png")
    return


def test_set_predict(model, X_test, Y_test, ident, encoder, showplot = False):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test).logits
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)

    Y_pred = [encoder.classes_[el] for el in Y_pred]
    Y_test = [encoder.classes_[el] for el in Y_test]

    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    print('Macro-F1 on own {1} set: {0}'.format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))
    if showplot:
        # get the classnames from encoder
        classnames = encoder.classes_
        matrix = calculate_confusion_matrix(Y_test, Y_pred, classnames)
        plot_confusion_matrix(matrix)
    return Y_pred, Y_test


def write_preds(X_test, Y_test, Y_pred, filename):
    """Write test predictions along with inputs and expected outputs
    Args:
        X_test (List): Text test sentences
        Y_test (List): Labels of test dataset
        Y_pred (List): Labels predicted
    """
    txtt = []
    for x, yt, yprd in zip(X_test, Y_test, Y_pred):
        txtt.append("\t".join([x,yt,yprd]))

    with open(filename, "w") as fp:
        fp.write("\n".join(txtt))


def read_data(args):
    # Read in the data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    return X_train, Y_train, X_dev, Y_dev

def vectorize_inputtext(args, tokenizer, X_train, X_dev):
    # Transform words to indices using a vectorizer
    tokens_train = tokenizer(X_train, padding=True, max_length=args.max_seq_len,
    truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=args.max_seq_len,
    truncation=True, return_tensors="np").data
    return tokens_train, tokens_dev

def numerize_labels(Y_train, Y_dev):
    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    return encoder, Y_train_bin, Y_dev_bin

def read_testdata_andvectorize(args, tokenizer, encoder):
    # Read in test set and vectorize
    X_test, Y_test = read_corpus(args.test_file)
    tokens_test = tokenizer(X_test, padding=True, max_length=args.max_seq_len,
    truncation=True, return_tensors="np").data
    Y_test_bin = encoder.fit_transform(Y_test)
    return X_test, Y_test, tokens_test, Y_test_bin

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)
    # Read in the data and embeddings
    X_train, Y_train, X_dev, Y_dev = read_data(args)
    # Create model
    model, tokenizer = create_model(args, Y_train, learning_rate = args.learning_rate, 
                                    lm = args.langmodel_name, batchsize = args.batch_size,
                                    num_epochs = args.num_epochs)
    tokens_train, tokens_dev = vectorize_inputtext(args, tokenizer, X_train, X_dev)
    encoder, Y_train_bin, Y_dev_bin = numerize_labels(Y_train, Y_dev)
    # Train the model
    model = train_model(model, tokens_train, Y_train_bin, tokens_dev,
                        Y_dev_bin, encoder= encoder ,batchsize = args.batch_size,
                        num_epochs = args.num_epochs)
    # Do predictions on specified test set
    if args.test_file:
        X_test, Y_test, tokens_test, Y_test_bin = read_testdata_andvectorize(args, tokenizer, encoder)
        # Finally do the predictions
        Y_pred, Y_test = test_set_predict(model, tokens_test, Y_test_bin,
                        "test", encoder, showplot=True)
        write_preds(X_test, Y_test, Y_pred, args.test_file+"predictions.txt")

if __name__ == '__main__':
    main()

