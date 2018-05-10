from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.optimizers import Adam, RMSprop

import numpy as np
import random
import os
import glob
import re
import codecs
import time
from datetime import timedelta

# Loads the file `filename` and returns:
# `vocabulary` : the set of all 'words' (as determined by the split RE)
# `corpus' : the text as an ordered list.
# "The cat the dog chased" -> vocabulary = set(["the", "cat", "dog", "chased"]
# corpus = list(["the", "cat", "the", "dog", "chased"])
def load_single_file_word (filename) :
    text = codecs.open(filename, encoding='utf-8').read().lower()
    #corpus = re.split('(\W+)', text)
    corpus = re.split('\W+', text)
    # strip spaces
    corpus = [x.strip() for x in corpus if x.strip() != ""]
    vocabulary = set(corpus)
    return (vocabulary, corpus)


def load_multi_file_word (path) :
    excludes = [ " ", "\n" , "\"", "\'", ]
    # Markdown only
    filenames = list(f for f in glob.iglob(path + '**/*.md', recursive=True))
    corpus = []
    for f in filenames :
        if len(corpus) > 100000 :
            return (set(corpus), corpus)
        text = codecs.open(f, encoding='utf-8').read().lower()
        split = re.split('(\W+)', text)
        stripped = [x.strip() for x in split if x.strip() != ""]
        filtered = filter(lambda s : s not in excludes, stripped)
        corpus.extend(filtered)
    vocabulary = set(corpus)
    return (vocabulary, corpus)

def load_single_file_char (filename) :
    print("Loading {0}".format(filename))
    excludes = [ '\n', '\t' ]
    corpus = list(codecs.open(filename, encoding='utf-8').read().lower())
    filtered = list(filter(lambda c : c not in excludes, corpus))
    vocabulary = set(filtered)
    return (vocabulary, filtered)

# Takes a list and returns `training_data` and `validation_data` where validation_data consists of
# spans of length `sequence_length`
def split_data(source_data, sequence_length = 15, validation_pct = 0.15) :
    validation_spans = int(validation_pct * len(source_data) // sequence_length)
    print("Removing {0} spans of length {1} from source of size {2}".format(validation_spans, sequence_length, len(source_data)))
    # Deep copy of corpus
    training_data = source_data[:]
    validation_data = []
    for i in range(validation_spans) :
        training_spans = len(training_data) // sequence_length
        span_index = random.randint(0, training_spans)
        start_index = span_index * sequence_length
        end_index = start_index + sequence_length
        slice = training_data[start_index : end_index]
        validation_data.extend(slice)
        # Removes the range from the training corpus
        training_data = training_data[:start_index] + training_data[end_index:]
    return (training_data, validation_data)


# Returns two dictionaries that map between int <-> T (whatever T is in categories : list<T>)
def category_index_multimap (categories) :
    index_for_word = dict((c, i) for i, c in enumerate(categories))
    word_for_index = dict((i, c) for i, c in enumerate(categories))

    print("Created multimap {1}<->int of length {0}".format(len(index_for_word), type(word_for_index[0])))

    return (index_for_word, word_for_index)

# Returns the `batch_generator` function that is used to create `count` sequences from `data` list
def build_batch_generator (batch_size, sequence_len, category_size, index_for_category) :
    print("Creating batch generator with:")
    print("X of shape [{0},{1},{2}]".format(batch_size, sequence_len, category_size))
    print("y of shape [{0},{1}]".format(batch_size, category_size))

    def batch_generator(data, count) :
        """Generate batches for training"""
        while True:  # Outer infinite loop: always use the same generator
            for batch_ix in range(count):
                X = np.zeros((batch_size, sequence_len, category_size))
                y = np.zeros((batch_size, category_size))

                batch_offset = batch_size * batch_ix

                for sample_ix in range(batch_size):
                    sample_start = batch_offset + sample_ix
                    for s in range(sequence_len):
                        X[sample_ix, s, index_for_category[data[sample_start + s]]] = 1
                    y[sample_ix, index_for_category[data[sample_start + s + 1]]] = 1

                yield X, y
    return batch_generator


def build_model(lstm_layer_count, hidden_layers_dim, dropout_pct, sequence_length, category_size, learning_rate = 0.001) :
    print('Building model: {0} LSTM layers of length {1} accepting sequences of {2} elements from among {3} categories, dropping out {4}'.format(
        lstm_layer_count, hidden_layers_dim, sequence_length, category_size, dropout_pct
    ))
    model = Sequential()
    for i in range(lstm_layer_count):
        model.add(
            LSTM(
                hidden_layers_dim,
                return_sequences=True if (i != (lstm_layer_count - 1)) else False,
                input_shape=(sequence_length, category_size)
            )
        )
        model.add(Dropout(dropout_pct))

    # model.add(BatchNormalization())
    model.add(Dense(category_size))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def reweight_distribution(original, temperature=0.5) :
    distribution = np.log(original)
    exp_dist = np.exp(distribution)
    return exp_dist / np.sum(exp_dist)

def seed (sequence, maxlen) :
    start_index = random.randint(0, len(sequence) - maxlen - 1)
    subsequence = sequence[start_index: start_index + maxlen]
    print ("Seed is '{0}".format(subsequence))
    return subsequence


# int -> int -> (T -> int) -> (model -> T seq -> T seq)
# Partially-apply parameters, returning the function that creates a new sequence from a model and a sequence
def build_choose_one_fn (sequence_length, vocabulary_size, value_to_index, index_to_value, temperature) :
    def next_fn (model, input_sequence) :
        x = np.zeros((1, sequence_length, vocabulary_size))
        trimmed_input = input_sequence[:sequence_length]
        for i, el in enumerate(trimmed_input) :
            if not el in value_to_index :
                print("WHOA! Trying to find an index for value '{0}' and it's missing".format(el))
                print("Trimmed input was {0}".format(trimmed_input))
                exit(1)
            value_index = value_to_index[el]
            # print ("i is {0}, el is {1}, ix is {2}".format(i, el, value_index))
            x[0, i, value_index] = 1.
        preds = model.predict(x, batch_size=1, verbose = 0)
        next_index = reweight_distribution (preds[0,:], temperature=temperature)
        next_value = index_to_value[next_index]
        return next_value
    return next_fn


# Create a new sequence of length `count` by repeatedly choosing a next element and then
def predict_sequence_recursive (model, choose_one_fn, sequence, count=0) :
    if count == 0 :
        return sequence
    next_val = choose_one_fn (model, sequence)
    # print(next_val)
    # print("Typeof sequence '{0}' is {1}".format(sequence, type(sequence)))
    sequence.append(next_val)
    return predict_sequence_recursive(model, choose_one_fn, sequence, count - 1)


def predict_sequence (model, sequence_length, vocabulary_size, value_to_index, index_to_value,temperature, seed) :
    choose_one_fn = build_choose_one_fn(sequence_length, vocabulary_size, value_to_index, index_to_value, temperature)
    seq = predict_sequence_recursive(
        model=model,
        choose_one_fn=choose_one_fn,
        sequence = seed,
        count = sequence_length
    )
    return seq

def predict_sentence_by_predicting_words (model, seed, sequence_length, vocabulary_size, value_to_index, index_to_value, temperature) :
    seq = predict_sequence(
        model=model,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        value_to_index=value_to_index,
        index_to_value=index_to_value,
        temperature=temperature,
        seed = seed)
    return ' '.join(seq)

def predict_sentence_by_predicting_chars (model, seed, sequence_length, vocabulary_size, value_to_index, index_to_value, temperature) :
    seq = predict_sequence(model, sequence_length, vocabulary_size, value_to_index, index_to_value, temperature, seed)
    return ''.join(seq)

def batch_count(list_len, sequence_len, batch_size) :
    return (list_len - sequence_len) // batch_size


def callbacks(checkpoint_save_weights_period, val_loss_patience) :
    # Callbacks
    filepath = "./epoch{epoch:02d}-loss{loss:.4f}_weights"

    checkpoint = ModelCheckpoint(
        filepath,
        save_weights_only=True,
        save_best_only=True,
        period=checkpoint_save_weights_period
    )

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=val_loss_patience)

    callbacks_list = [checkpoint, early_stopping]
    return callbacks_list

def hyper_parameters() :
    # Trying to find some reasonable hyper-params for character-level
    sequence_len = 60
    validation_pct = 0.10
    # Batch_size = 1 = stochastic
    batch_size = 15 # A line of output = samples_per_epoch / batch_size
    epochs_per_iteration = 50
    max_epochs = 1000
    lstm_layer_count = 2
    hidden_layer_dim = 512
    dropout_pct = 0.2
    checkpoint_save_weights_period = 5
    val_loss_patience = 3
    learning_rate = 0.01


    return (sequence_len,
    validation_pct,
    batch_size,
    epochs_per_iteration,
    max_epochs,
    lstm_layer_count,
    hidden_layer_dim,
    dropout_pct,
    checkpoint_save_weights_period,
    val_loss_patience,
    learning_rate
    )


def main() :

    (sequence_len,
     validation_pct,
     batch_size,
     epochs_per_iteration,
     max_epochs,
     lstm_layer_count,
     hidden_layer_dim,
     dropout_pct,
     checkpoint_save_weights_period,
     val_loss_patience,
     learning_rate
     ) = hyper_parameters()

    #(vocabulary, corpus) = load_single_file_word("shakespeare.txt")
    (vocabulary, corpus) = load_single_file_char("nietzsche.txt")

    print("Type: categories are of type : {0}".format(type(corpus[0])))
    print("Volume: categories has {0} values, corpus has {1} elements".format(len(vocabulary), len(corpus)))

    (index_for_word, word_for_index) = category_index_multimap(vocabulary)

    (training_data, validation_data) = split_data(corpus, sequence_len, validation_pct)
    print("Training set: {0} sequences, Validation set: {1} sequences. Sequence = {2} elements".format(len(training_data),
                                                                                                    len(validation_data),
                                                                                                    sequence_len))

    model = build_model(
        lstm_layer_count=lstm_layer_count,
        hidden_layers_dim=hidden_layer_dim,
        dropout_pct=dropout_pct,
        sequence_length=sequence_len,
        category_size=len(vocabulary),
        learning_rate=learning_rate,
    )

    # Test generation -- this will be crap
    test_sentence = predict_sentence_by_predicting_chars(
        model=model,
        seed=['h', 'e', 'l', 'l', 'o'],
        sequence_length=sequence_len,
        vocabulary_size=len(vocabulary),
        value_to_index=index_for_word,
        index_to_value=word_for_index,
        temperature=1.0
    )

    print("Test sentence on an untrained model: {0}".format(test_sentence))

    callbacks_list = callbacks(checkpoint_save_weights_period, val_loss_patience)

    samples_per_epoch = len(training_data) // sequence_len
    steps_per_epoch = samples_per_epoch // batch_size
    training_iterations = max_epochs // epochs_per_iteration

    print("Samples_per_epoch {0}".format(samples_per_epoch))
    print("Steps per epoch {0}".format(steps_per_epoch))
    print("Samples per step (is this minibatches / epoch?) {0}".format(samples_per_epoch/steps_per_epoch))
    if steps_per_epoch == 0 :
        print("Something failed -- steps_per_epoch = 0")
        print("Manipulate samples_per_epoch and batch_size such that steps_per_epoch is a whole number")
        exit(1)
    print("Epochs per iteration {0}".format(epochs_per_iteration))
    print("Training iterations = {0}".format(training_iterations))

    train_batch_count = batch_count(len(training_data), sequence_len, batch_size)
    val_batch_count = batch_count(len(validation_data), sequence_len, batch_size)

    print("Confirming val_batch_count as {0}".format(val_batch_count))

    batch_generator = build_batch_generator(batch_size, sequence_len, len(vocabulary), index_for_word)
    # train the model, output generated ordered_words after each iteration
    training_time_start = time.time()

    for iteration in range(training_iterations):
        history = model.fit_generator(
            generator = batch_generator(training_data, train_batch_count),
            steps_per_epoch = steps_per_epoch,
            epochs = epochs_per_iteration,
            verbose = 1,
            callbacks = callbacks_list,
            validation_data=batch_generator(validation_data, val_batch_count),
            validation_steps=val_batch_count,
            max_queue_size=1,
            shuffle=True
        )

        seed_sequence = seed(training_data, sequence_len)

        print("----")
        elapsed = time.time() - training_time_start
        print("Epoch {0}, training time: {1}".format(iteration * epochs_per_iteration, timedelta(seconds=elapsed)))
        for temperature in [0.8, 1.0, 1.2]:
            generated_sentence = predict_sentence_by_predicting_chars(
                model=model,
                seed=seed_sequence,
                sequence_length=sequence_len,
                vocabulary_size=len(vocabulary),
                value_to_index=index_for_word,
                index_to_value=word_for_index,
                temperature=temperature
            )
            clear = "[{0}]{1}".format(generated_sentence[:sequence_len], generated_sentence[sequence_len:])
            msg = "Temperature {0} : {1}".format(temperature, clear)
            print(msg)

    #model.save_weights('weights')

main()