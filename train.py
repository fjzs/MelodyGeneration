# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import keras as K
import os
import numpy as np
import tensorflow as tf
from preprocess import ENCODED_SONGS_FOLDER_PATH, DIRECTORY
from file_utils import load_plain_file
from midi_index_utils import VOCABULARY_SIZE

# ------------------------- Dataset parameters ---------------------------------#
SEQUENCE_LENGTH_TX = 128

# ------------------------- Training parameters --------------------------------#
NUM_HIDDEN_UNITS_PER_LAYER = [64] #256
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.01
BATCH_SIZE = 256
MODEL_FILE = DIRECTORY + "/model.h5"
DROPOUT_RATE_PER_LAYER = [0.2] #list containing the drop-out rate per layer
# ------------------------------------------------------------------------------#


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
    output_units = VOCABULARY_SIZE
    
    # Create the model architecture
    input_shape = (None, output_units) # None is to have flexibility for the length generation
    input_layer = K.layers.Input(shape = input_shape) 
    x = K.layers.LSTM(units = num_units_per_layer[0])(input_layer)
    x = K.layers.Dropout(rate = dropout_per_layer[0])(x)
    output = K.layers.Dense(output_units, activation="softmax")(x)
    model = K.Model(input_layer, output)

    # Compile model
    my_optimizer = K.optimizers.Adam(learning_rate)
    model.compile(loss = loss_function, optimizer = my_optimizer, metrics=["accuracy"])
    
    print("\nModel summary")
    model.summary()

    return model

def generate_dataset(encodings_directory: str = ENCODED_SONGS_FOLDER_PATH, tx: int = SEQUENCE_LENGTH_TX):
    """
    Create X and Y data samples for training. Each sample is a sequence.
    For instance, if sequence_length = 3 and for an input as X = [1,2,3,4,5,6,7,8,9], 
    then some training examples would be:
        [1,2,3] -> 4
        [2,3,4] -> 5
        [3,4,5] -> 6
    *** Then this information is one-hot-encoded    
    
    Arguments:
        encodings_directory (str): where to find the encoded files
        tx (int): the length of each sequence

    Returns:
        X (ndarray): Training inputs,  shape = (m, Tx, vocabulary size)
        Y (ndarray): Training targets, shape = (m, )
    """
    
    # Initialize outputs
    X = []
    Y = []
    
    print(f"Generating dataset with Tx = {tx}")
    
    # Load encoded songs (text files)
    for path, _, files in os.walk(encodings_directory):
        for i,file in enumerate(files):
            file_name, file_extension = os.path.splitext(file)
            
            #Load just some files
            if i == 1:
            
                # Just consider text files
                if file_extension == ".txt":            
                    file_path = os.path.join(path, file) 
                    encoded_song = load_plain_file(file_path)
                    indices_list_str = encoded_song.split(',') #these are string elements
                    indices_list_int = [int(i) for i in indices_list_str] #cast as int
                    
                # Generate the training sequences
                # if we have 5 symbols: [1,2,3,4,5] and sequence length is 3, then we have:
                # [1,2,3] -> 4, [2,3,4] -> 5   => 2 sequences
                num_examples = len(indices_list_int) - tx
                print(f"\tFile #{i+1} has {num_examples} examples")
                if num_examples > 0:            
                    for i in range(num_examples):
                        
                        # List of values from i ... i + Tx - 1
                        training_example_x = np.array(indices_list_int[i : i + tx], dtype = np.int32)
                        
                        # Value with index = i + Tx
                        training_example_y = np.array(indices_list_int[i + tx], dtype = np.int32)     
                        
                        # Append to dataset
                        X.append(training_example_x) 
                        Y.append(training_example_y)
    
    # One-Hot encode the sequences
    X = K.utils.to_categorical(X, num_classes = VOCABULARY_SIZE)
    Y = np.array(Y, dtype = np.int32)
    print(f"The dataset has a total of {X.shape[0]} examples")
    
    return X, Y    
    
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
    
    # Generate dataset
    X, Y = generate_dataset()
    m = X.shape[0]    

    # Check GPU availability
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("GPU not found")
    
    # Build the model
    model = build_model()

    # train the model
    total_epochs = int(m/BATCH_SIZE) 
    history = model.fit(X, Y, epochs = total_epochs, batch_size = BATCH_SIZE, verbose = 1)
    print(type(history))
    print(history)
    
    # save the model to not start everytime from scratch
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()


