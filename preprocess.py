# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import os
import music21 as m21
import json
import keras as K
import numpy as np
import collections

# ----------------- Directories constants --------------------------------#
DIRECTORY = "test" #slashes like '/' for subdirectories
SONGS_PATH = DIRECTORY + "/songs"
ENCODED_SONGS_FOLDER_PATH = DIRECTORY + "/encoded_songs"
SINGLE_FILE_DATASET_PATH = DIRECTORY + "/single_file_encoded_songs"
MAPPING_SYMBOL_TO_INDEX_PATH = DIRECTORY + "/mapping_symbol_to_index.json"
MAPPING_INDEX_TO_SYMBOL_PATH = DIRECTORY + "/mapping_index_to_symbol.json"
# ------------------------------------------------------------------------#

# ----------------- Preprocessing constants ------------------------------#
SEQUENCE_LENGTH = 64 # Length of the sequence input of the LSTM model
ACCEPTABLE_DURATIONS_MULTIPLE = 0.25 #every duration multiple of a 16th note
# ------------------------------------------------------------------------#


def load_songs(dataset_path):
    """
    Loads all pieces in dataset using music21.
    
    -param dataset_path (str): Path to dataset
    -return songs (list of m21 streams): List containing all pieces
    """
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        print(f"\tnumber of available songs: {len(files)}")
        for file in files:
            song = m21.converter.parse(os.path.join(path, file))
            songs.append(song) #Music21 Score                
       
    return songs

def has_acceptable_durations(song, minimum_duration_multiple = ACCEPTABLE_DURATIONS_MULTIPLE):
    """
    Returns True if piece has all acceptable duration, False otherwise.
    The song duration has to be multiple of 
    
    Arguments:
    - minimum_duration_multiple (float): minimum multiplier of duration
    - song (m21 stream)
    
    Returns:
    - true or false
    """
    #Looks at all of the notes M21 Objects   
    print("\tduration histogram:")
    duration_to_frequency = {}
    
    # Creates dictionary of durations
    for note in song.flat.notesAndRests:
        duration = note.duration.quarterLength      
        if duration not in duration_to_frequency.keys():
             duration_to_frequency[duration] = 0
        duration_to_frequency[duration] += 1   
    ordered_key_dict = collections.OrderedDict(sorted(duration_to_frequency.items()))
        
    # checks if duration is ok
    is_acceptable = True
    for dur, freq in ordered_key_dict.items():
        is_dur_ok = (dur % minimum_duration_multiple == 0)
        is_acceptable = is_acceptable and is_dur_ok
        print(f"\t\td = {dur} appears {freq} times, acceptable: {is_dur_ok}")
    
    print(f"\t\tacceptable durations: {is_acceptable}")
    return is_acceptable

def transpose(song):
    """
    Transposes song to C maj/A min. This is done to facilitate the learninig process afterwards, 
    otherwise, the model would have to learn all the 24 keys and thus need more data. It is better 
    to just learn 2 keys, C major and A minor.
    -param piece (m21 stream): Piece to transpose
    -return transposed_song (m21 stream)
    """

    # get key from the song
    parts_list = song.getElementsByClass(m21.stream.Part)
    first_part = parts_list[0].getElementsByClass(m21.stream.Measure) #getting all the measures
    key = first_part[0][4] # the key is usually stored in the index = 4

    # If this object is not a m21.Key object, estimate it
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    print(f"\tkey: {key.name}")    
    
    # get interval for transposition, i.e., if key = Bmaj, how much to move from Bmaj to Cmaj?
    if key.mode == "major":    #if the key is major calculate how to move to Cmaj
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":  #if the key is minor calculate how to move to Amin
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song, time_step = 0.25):
    """
    Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:
    
    C4 note for 1 duration would be:
    [60, "_", "_", "_"]
    Pitch Symbol = 60
    The note lasts 4 quarter lengths
        
    Arguments:
    -song (m21 stream): Piece to encode
    -time_step (float): Duration of each time step in quarter length

    Returns:
    None
    """
        
    encoded_song = []

    # Flattening all the elements of the song and consider just (A) notes and (B) rests
    for event in song.flat.notesAndRests:

        # (A) handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 
        # (B) handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note or the rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # If it's the first time we see a note/rest, let's encode it. "60" of [60, "_", "_", "_"]
            # Otherwise, it means we're carrying the same symbol in a new time step "_" of [60, "_", "_", "_"]
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    # join all the items by a space, first mape them all to be a str type (map function)
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song

def preprocess(songs_path = SONGS_PATH):
    """
    1. Loads the songs
    2. Eliminate songs with non acceptable durations
    3. Transpose the to Cmaj/Amin
    4. Encode each song with a music time series representation
    5. Save songs to text file
    
    Arguments:
        songs_path (str): path of the songs to be loaded
    
    Returns:
        None    
    """
        
    # 1. Loads the songs
    print("Loading songs...")
    songs = load_songs(songs_path)
          
    # Enumerate song one by one, indexing by i
    saved_songs = 0
    for i, song in enumerate(songs):

        print(f"\nanalyzing song #{i+1}")        

        # 2. Eliminate songs with non acceptable durations
        if not has_acceptable_durations(song):
            print("\tsong not encoded...")
            continue #skip the song

        # 3. Transpose the song to Cmaj or Amin key
        song = transpose(song)

        # 4. Encode each song with music time series representation
        encoded_song = encode_song(song)

        # Create folder if not existant
        if not os.path.exists(ENCODED_SONGS_FOLDER_PATH):
            os.makedirs(ENCODED_SONGS_FOLDER_PATH)

        # Save songs to text file
        file_name = "encoded_song_"+ str(i+1)
        file_path = ENCODED_SONGS_FOLDER_PATH +"/"+file_name
        with open(file_path, "w") as fp:
            fp.write(encoded_song)
            saved_songs += 1
            print(f"\tsaved as {file_name}")
        
    print(f"\nnumber of encoded songs: {saved_songs}")

def load(file_path):
    """
    Reads the encoded path for a song
    
    Arguments:
    - file_path (str): path to the encoded song
    
    Returns:
    - song (str): the str which has the encoding
    """
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path = ENCODED_SONGS_FOLDER_PATH, 
                               file_dataset_path = SINGLE_FILE_DATASET_PATH, 
                               sequence_length = SEQUENCE_LENGTH):
    """
    Generates a file combining all the encoded songs and adding new piece delimiters.

    Args:
        dataset_path (str): Path to folder containing the encoded songs.
        file_dataset_path (str): Path to file for saving songs in single file.
        sequence_length (int): # of time steps to be considered for training.

    Returns:
        songs (str): String containing all songs in dataset + delimiters.

    """
    
    print("\nCreating single file with all the encoded songs...")
    new_song_delimiter = "/ " * sequence_length # to identify the end of a song
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file) #this is the full path of the song
            song = load(file_path) # is a string like "60 _ _ _ 61 _ _ _ 62 _ 66 _ ..."
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp: #open the file in writing mode
        fp.write(songs)
        
    print("\tfile created")
    return songs

def create_mapping(songs):
    """
    Creates 2 dictionaries as a json file that maps both the symbols to indexes and viceversa
    
    Args:
        songs (str): String with all songs, such as "60 _ _ 63 _ 61 62 r ..."
                
    Returns:
        None
    """
    
    print("\nCreating mappings symbol -> index and viceversa")
    
    mappings_symbol_to_index = {}
    mappings_index_to_symbol = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs)) #the set eliminate duplicates symbols
    
    # create mappings
    print(f"\tthere are {len(vocabulary)} unique symbols. The elements are:")
    for i, symbol in enumerate(vocabulary):
        mappings_symbol_to_index[symbol] = i
        mappings_index_to_symbol[i] = symbol
        print(f"\t\t{symbol} <-> {i}")

    # save both dictionaries to a json file
    with open(MAPPING_SYMBOL_TO_INDEX_PATH, "w") as fp: #open the file in writing mode
        json.dump(mappings_symbol_to_index, fp, indent=4) #indent to easier reading
    
    with open(MAPPING_INDEX_TO_SYMBOL_PATH, "w") as fp: #open the file in writing mode
        json.dump(mappings_index_to_symbol, fp, indent=4) #indent to easier reading

    print("\tdictionaries created")

def convert_songs_to_int_list(songs):
    """
    Takes a symbols string separated by " " and returns it as a list of integers
    
    -param songs (str): lots og songs appended together as string symbols
    -return a list of integers
    """
    int_songs = []

    # load mappings
    with open(MAPPING_SYMBOL_TO_INDEX_PATH, "r") as fp: #read mode
        mappings_symbol_to_index = json.load(fp)

    # transform songs string to list of symbols
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings_symbol_to_index[symbol])

    return int_songs


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
    songs = load(SINGLE_FILE_DATASET_PATH)
    int_songs = convert_songs_to_int_list(songs)

    X = []
    Y = []

    # generate the training sequences
    # if we have 5 symbols: [1,2,3,4,5] and sequence length is 3, then we have:
    # [1,2,3] -> 4, [2,3,4] -> 5   => 2 sequences
    num_examples = len(int_songs) - sequence_length_Tx
    for i in range(num_examples):
        training_example_x = int_songs[i : i + sequence_length_Tx] # List of values from i ... i + Tx - 1
        training_example_y = int_songs[i + sequence_length_Tx]     # Value with index = i + Tx
        X.append(training_example_x) 
        Y.append(training_example_y)

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs)) #number of different values        
    X = K.utils.to_categorical(X, num_classes = vocabulary_size)
    Y = np.array(Y)

    return X, Y


def main():
    preprocess()
    songs = create_single_file_dataset()
    create_mapping(songs)
    print("\nPreprocessing finished!")

if __name__ == "__main__":
    main()


