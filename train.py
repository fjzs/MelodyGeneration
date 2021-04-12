# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import keras as K
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_HIDDEN_UNITS_PER_LAYER = [64] #256
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.01
EPOCHS = 5 #50
BATCH_SIZE = 8 #64
SAVE_MODEL_PATH = "model.h5"
DROPOUT_RATE_PER_LAYER = [0.2]


def build_model(output_units, num_units_per_layer, dropout_per_layer, my_loss, learning_rate):
    """
    Builds and compiles an LSTM model
    
    Arguments:
    - output_units (int): Num output units
    - num_units (list of int): Num of units in hidden layers
    - loss (str): Type of loss function to use
    - learning_rate (float): Learning rate to apply
    
    Returns:
    - model (tf model): Where the magic happens :D
    """

    # create the model architecture
    input_shape = (None, output_units) # None is to have flexibility for the length generation
    input_layer = K.layers.Input(shape = input_shape) 
    x = K.layers.LSTM(units = num_units_per_layer[0])(input_layer)
    x = K.layers.Dropout(rate = dropout_per_layer[0])(x)
    output = K.layers.Dense(output_units, activation="softmax")(x)
    model = K.Model(input_layer, output)

    # compile model
    my_optimizer = K.optimizers.Adam(learning_rate)
    model.compile(loss = my_loss, optimizer = my_optimizer, metrics=["accuracy"])
    model.summary()

    return model


def train(output_units = OUTPUT_UNITS, num_units = NUM_HIDDEN_UNITS_PER_LAYER, loss = LOSS, learning_rate = LEARNING_RATE):
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

    # generate the training sequences
    X, Y = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, DROPOUT_RATE_PER_LAYER, loss, LEARNING_RATE)

    # train the model
    history = model.fit(X, Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1)

    # save the model to not start everytime from scratch
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()


