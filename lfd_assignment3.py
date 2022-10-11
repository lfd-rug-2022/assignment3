#!/usr/bin/env python

''' Trains a LSTM model using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.

To run this file,

1. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

python lfd_assignment3.py

Running with the best hyperparams must give a macro F1-score of around 89/90.

Or if you want to change any of the arguments, please type

python lfd_assignment3.py --help
'''

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM, Dropout
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import TextVectorization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model

import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
tf.keras.utils.set_random_seed(1234)
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
    parser.add_argument("--show_confusionmatrix", default=True, type=bool,
                        help="Show confusion matrix of the model on test set")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="Maximum length of input sequence after BPE")
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


def create_model(args, Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    loss_function = 'categorical_crossentropy'
    optim = Adam(learning_rate=args.learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))

    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, 
                        embeddings_initializer=Constant(emb_matrix),trainable=True,
                        name="embedding_updatable"))
    # Here you should add LSTM layers (and potentially dropout)
    # raise NotImplementedError(f" Emb matrix {embedding_dim} Add LSTM layer(s) here")
    model.add(Dense(units=512, activation="relu", name="dense_512"))
    model.add(Dropout(0.5, name="dropout_0.5"))
    model.add(LSTM(512, recurrent_dropout=0.2, name="LSTM_512"))
    model.add(Dropout(0.5, name="dropout2_0.5"))
    # Ultimately, end with dense layer with softmax
    model.add(Dense(units=num_labels, activation="softmax", name=f"dense_{num_labels}"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file='model.png')
    return model


def train_model(args, model, X_train, Y_train, X_dev, Y_dev, encoder):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = args.batch_size
    epochs = args.num_epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    earlystopping = EarlyStopping(monitor='val_loss', patience=3)
    mckpt = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, 
              callbacks=[earlystopping, mckpt], batch_size=batch_size, 
              validation_data=(X_dev, Y_dev))
    
    # load the saved model
    model = load_model('best_model.h5')

    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev", encoder)
    return model


def test_set_predict(model, X_test, Y_test, ident, encoder, showplot = False):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)

    Y_pred = [encoder.classes_[el] for el in Y_pred]
    Y_test = [encoder.classes_[el] for el in Y_test]

    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    print('Macro F1 on own {1} set: {0}'.format(round(f1_score(Y_test, Y_pred, average = 'macro'), 3), ident))

    if showplot:
        # get the classnames from encoder
        classnames = encoder.classes_
        matrix = calculate_confusion_matrix(Y_test, Y_pred, classnames)
        plot_confusion_matrix(matrix)

def calculate_confusion_matrix(Y_test, y_pred, labels,):
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

def read_data_embeddings(args):
    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)
    return X_train, Y_train, X_dev, Y_dev, embeddings

def create_vectorizer(args, X_train, X_dev):
    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=args.max_seq_len)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    return vectorizer, voc

def numerize_labels(Y_train, Y_dev):
    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    return encoder, Y_train_bin, Y_dev_bin

def vectorize_inputtext(vectorizer, X_train, X_dev):
    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()
    return X_train_vect, X_dev_vect

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    X_train, Y_train, X_dev, Y_dev, embeddings = read_data_embeddings(args)
    vectorizer, voc = create_vectorizer(args, X_train, X_dev)
    emb_matrix = get_emb_matrix(voc, embeddings)
    # Create model
    model = create_model(args, Y_train, emb_matrix)
    encoder, Y_train_bin, Y_dev_bin = numerize_labels(Y_train, Y_dev)
    X_train_vect, X_dev_vect = vectorize_inputtext(vectorizer, X_train, X_dev)
    # Train the model
    model = train_model(args, model, X_train_vect, Y_train_bin,
                        X_dev_vect, Y_dev_bin, encoder)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin,
                         "test", encoder, showplot=True)

if __name__ == '__main__':
    main()
