# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import os
import json
import keras as K
import numpy as np
import collections
from file_utils import create_plain_file, load_plain_file
from MidiIndexUtils import readMidi, midiToIdxs


# ----------------------------- CONSTANTS --------------------------------#
DIRECTORY = "maestro_v3" #slashes like '/' for subdirectories
SONGS_PATH = DIRECTORY + "/1. initial songs"
ENCODED_SONGS_FOLDER_PATH = DIRECTORY + "/2. encoded songs"
ALLOWED_EXTENSIONS = [".mid", ".midi"]

# ------------------------------------------------------------------------#

def load_songs(dataset_path):
    """
    Loads all mid files in the specified dataset using music21.
    
    Arguments:
        dataset_path (str): Path to dataset
    
    Returns:
        List of tuples (file_name, midi file)
    """
    file_name_and_midi_files = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        loaded_files = 0
        for file in files:
            #https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python
            file_name, file_extension = os.path.splitext(file) 
            if file_extension in ALLOWED_EXTENSIONS:
                print(f"\tLoading {file_name}...")
                full_path = os.path.join(path, file)
                mf = readMidi(full_path)                
                tuple_= (file_name, mf)
                file_name_and_midi_files.append(tuple_)
                loaded_files += 1
    
    print(f"Number of loaded files: {loaded_files} out of {len(files)}")
    return file_name_and_midi_files

def preprocess(songs_path = SONGS_PATH):
    """
    1. Loads the songs
    2. Encode each song with a music time series representation
    3. Save songs to text file
    
    Arguments:
        songs_path (str): path of the songs to be loaded
    
    Returns:
        None    
    """
        
    # 1. Loads the songs
    print("Loading songs...")
    file_name_and_midi_files = load_songs(songs_path)
    quantity_songs = len(file_name_and_midi_files)
    max_digits = len(str(quantity_songs))
          
    # Enumerate song one by one, indexing by i
    saved_songs = 0
    for i, songName_midiFile in enumerate(file_name_and_midi_files):
        
        # retrieve the info from the tuple
        song_name, midi_file = songName_midiFile 
                
        # 2. Encode each song with music time series representation
        list_indices = midiToIdxs(midi_file)
        string_indices = ",".join(map(str, list_indices))
        
        # 3. Save encoded song to a text file
        encoded_file_name = "encoded song " + str(i).zfill(max_digits) + " " + song_name
        create_plain_file(ENCODED_SONGS_FOLDER_PATH, encoded_file_name, string_indices, "txt")
                            
        # Update counter
        saved_songs += 1        
        
    print(f"\nNumber of encoded songs: {saved_songs}")

def generate_training_sequences(sequence_length_Tx):
    """
    Create X and Y data samples for training. Each sample is a sequence.
    For instance, if sequence_length = 3 and for an input as X = [1,2,3,4,5,6,7,8,9], 
    then some training examples would be:
        [1,2,3] -> 4
        [2,3,4] -> 5
        [3,4,5] -> 6
    *** Then this information is one-hot-encoded    
    
    Arguments:
        sequence_length_Tx (int): Length of each sequence. With a quantisation at 16th notes, 
        64 notes equates to 4 bars
    
    Returns:
        X (ndarray): Training inputs,  shape = (m, Tx, vocabulary size)
        Y (ndarray): Training targets, shape = (m, )
    """

    # load songs from encoded single file and map them to int
    encoded_single_file = None
    song_encoded_as_indexes = None

    # Initialize outputs
    X = []
    Y = []

    # generate the training sequences
    # if we have 5 symbols: [1,2,3,4,5] and sequence length is 3, then we have:
    # [1,2,3] -> 4, [2,3,4] -> 5   => 2 sequences
    num_examples = len(song_encoded_as_indexes) - sequence_length_Tx
    for i in range(num_examples):
        training_example_x = song_encoded_as_indexes[i : i + sequence_length_Tx] # List of values from i ... i + Tx - 1
        training_example_y = song_encoded_as_indexes[i + sequence_length_Tx]     # Value with index = i + Tx
        X.append(training_example_x) 
        Y.append(training_example_y)

    # one-hot encode the sequences
    vocabulary_size = len(set(song_encoded_as_indexes)) #number of different values        
    X = K.utils.to_categorical(X, num_classes = vocabulary_size)
    Y = np.array(Y)

    return X, Y

def main():
    preprocess()
    #songs = create_single_file_dataset()
    #create_mapping(songs)
    print("\nPreprocessing finished!")

if __name__ == "__main__":
    main()


