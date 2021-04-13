# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import keras as K
from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_SYMBOL_TO_INDEX_PATH, DIRECTORY
import json

# ------------------------- Training parameters --------------------------------#
NUM_HIDDEN_UNITS_PER_LAYER = [64] #256
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.01
EPOCHS = 36 #50
BATCH_SIZE = 128 #64
SAVE_MODEL_PATH = DIRECTORY + "/model.h5"
DROPOUT_RATE_PER_LAYER = [0.2] #list containing the drop-out rate per layer
# ------------------------------------------------------------------------------#


def get_vocabulary_length(mapping_path = MAPPING_SYMBOL_TO_INDEX_PATH):
    """
    Counts the length of the vocabulary

    Args:
        mapping_path (sr, optional): dictionary path. Defaults to MAPPING_SYMBOL_TO_INDEX_PATH.

    Returns:
        vocabulary_size (int)

    """
    
    # load mappings
    with open(MAPPING_SYMBOL_TO_INDEX_PATH, "r") as fp: #read mode
        mappings_symbol_to_index = json.load(fp)
        vocabulary_size = len(mappings_symbol_to_index) 
        return vocabulary_size


def build_model(num_units_per_layer = NUM_HIDDEN_UNITS_PER_LAYER, 
                dropout_per_layer = DROPOUT_RATE_PER_LAYER, 
                loss_function = LOSS, 
                learning_rate = LEARNING_RATE):
    """
    Builds and compiles an LSTM model
    
    Arguments:
    - output_units (int): Num output units
    - num_units (list of int): Num of units on each hidden layers
    - loss (str): Type of loss function to use
    - learning_rate (float): Learning rate to apply
    
    Returns:
    - model (tf model): Where the magic happens :D
    """
    
    # Calculate the number of output units as the vocabulary size
    output_units = get_vocabulary_length()
    
    # create the model architecture
    input_shape = (None, output_units) # None is to have flexibility for the length generation
    input_layer = K.layers.Input(shape = input_shape) 
    x = K.layers.LSTM(units = num_units_per_layer[0])(input_layer)
    x = K.layers.Dropout(rate = dropout_per_layer[0])(x)
    output = K.layers.Dense(output_units, activation="softmax")(x)
    model = K.Model(input_layer, output)

    # compile model
    my_optimizer = K.optimizers.Adam(learning_rate)
    model.compile(loss = loss_function, optimizer = my_optimizer, metrics=["accuracy"])
    
    print("\nModel summary")
    model.summary()

    return model


def train():
    """
    Train and save TF model.
    
    Arguments:
    - output_units (int): Num output units
    - num_units (list of int): Num of units in hidden layers
    - loss (str): Type of loss function to use
    - learning_rate (float): Learning rate to apply
    
    Returns:
    None
    """
    
    print("Training the model")
    
    # generate the training sequences
    X, Y = generate_training_sequences(SEQUENCE_LENGTH)
    print(f"\ttraining examples = {X.shape[0]}")

    # build the network
    model = build_model()

    # train the model
    history = model.fit(X, Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1)

    # save the model to not start everytime from scratch
    model.save(SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train()


