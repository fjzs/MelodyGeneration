# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import os
import music21 as m21
import json
import keras as K
import numpy as np
import collections
from file_utils import create_plain_file, load_plain_file, create_midi_file_from_encoded_text_file
from fractions import Fraction


# ----------------- Directories constants --------------------------------#
DIRECTORY = "mozart" #slashes like '/' for subdirectories
SONGS_PATH = DIRECTORY + "/1. initial songs"
MODIFIED_SONGS_PATH = DIRECTORY + "/2. pre-encoded songs"
ENCODED_SONGS_FOLDER_PATH = DIRECTORY + "/3. encoded songs"
SINGLE_FILE_DATASET_PATH = DIRECTORY + "/single_file_encoded_songs.txt"
MAPPING_SYMBOL_TO_INDEX_PATH = DIRECTORY + "/mapping_symbol_to_index.json"
MAPPING_INDEX_TO_SYMBOL_PATH = DIRECTORY + "/mapping_index_to_symbol.json"
# ------------------------------------------------------------------------#

# ----------------- Preprocessing constants ------------------------------#
ALLOWED_EXTENSIONS = [".krn", ".mid"]
SEQUENCE_LENGTH = 128 # Length of the sequence input of the LSTM model

# https://web.mit.edu/music21/doc/moduleReference/moduleDuration.html
# In terms of a quarter length, thus, duration of 1 time step = duration of 1/TIME_STEP_DURATION quarter note
TIME_STEP_DURATION = Fraction(1,12) # measure unit = [quarter_length_duration / time_step]
# ------------------------------------------------------------------------#

def load_songs(dataset_path):
    """
    Loads all pieces in the specified dataset using music21.
    
    Arguments:
        dataset_path (str): Path to dataset
    
    Returns:
        List of tuples (file_name, m21.Score)
    """
    filename_and_songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        print(f"Number of available files: {len(files)}")
        for file in files:
            file_name, file_extension = os.path.splitext(file) #https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python
            if file_extension in ALLOWED_EXTENSIONS:
                print(f"\nLoading {file_name}...")
                song = m21.converter.parse(os.path.join(path, file)) #type = Music21 Score
                tuple_= (file_name, song)
                filename_and_songs.append(tuple_)          
       
    return filename_and_songs

        
def get_simple_list_general_notes(stream):
    """
    Get a list of (note type, duration) in the M21 stream object

    Args:
        stream (Music21 Stream): list of general notes objects.

    Raises:
        Exception: If the class is unknown for any element of the stream.

    Returns:
        simple_list (list of (str,float)): name and duration of each note.

    """
    
    simple_list = []
        
    for i, element in enumerate(stream):
        
        name = ""
        duration = str(element.quarterLength)
                
        if isinstance(element, m21.note.Rest):   # handle rest
            name = "rest"
        elif isinstance(element, m21.note.Note): # handle note
            name = element.nameWithOctave
        elif isinstance(element, m21.chord.Chord): # handle chord
            name = element.pitchedCommonName
        else:
            raise Exception("Class unknown for element e {e}")
        
        simple_list.append((name,duration))
    
    return simple_list

def has_acceptable_durations(song, time_step_duration = TIME_STEP_DURATION):
    """
    Returns True if piece has all acceptable duration, False otherwise.
    The song duration has to be multiple of 
    
    Arguments:
    - song (m21 stream)
    - minimum_duration_multiple (float): minimum multiplier of duration
        
    Returns:
    - true or false
    """
    #Looks at all of the notes M21 Objects   
    print("\n\tDuration histogram for each part:")
    
    # Look at each part    
    parts_list = song.getElementsByClass(m21.stream.Part)
    notes_list_per_part = {}
    acceptable_by_part = {}
    
    for i, p in enumerate(parts_list):
        
        part_name = p.partName
        acceptable_by_part[p] = True
        print(f"\n\t\tPart #{i+1}, name = {part_name}")
        print("\t\tduration histogram:")    
    
        # Creates dictionary of durations for each part
        duration_to_frequency = {} #dictionary duration -> frequency
        notes_and_rests = p.flat.notesAndRests        
        notes_list_per_part[part_name] = get_simple_list_general_notes(notes_and_rests)
        
        # Fill the notes dictionary
        for note in notes_and_rests:
            note_duration = note.duration.quarterLength            
            if note_duration not in duration_to_frequency.keys():
                 duration_to_frequency[note_duration] = 0
            duration_to_frequency[note_duration] += 1            
        ordered_dict_by_val = dict(sorted(duration_to_frequency.items(), key=lambda item: item[1]))
            
        # checks if duration is ok, that is, when the duration of the event can be written
        # as an integer multiple of our time_step        
        for dur, freq in ordered_dict_by_val.items():
            duration_time_steps = Fraction(dur/time_step_duration)
            is_duration_ok = duration_time_steps.denominator == 1
            acceptable_by_part[p] = acceptable_by_part[p] and is_duration_ok
            print(f"\t\t\td = {dur} appears {freq} times, acceptable: {is_duration_ok}")
                
    # check acceptabilty for all the parts
    is_acceptable = True
    for i, p in enumerate(acceptable_by_part.keys()):
        part_name = p.partName
        print(f"\n\tPart #{i+1}, name = {part_name} is acceptable: {acceptable_by_part[p]}")
        is_acceptable = is_acceptable and acceptable_by_part[p]
    
    print(f"\nSong is acceptable: {is_acceptable}")
    return is_acceptable

def transpose(song):
    """
    Transposes song to C maj/A min. This is done to facilitate the learninig process afterwards, 
    otherwise, the model would have to learn all the 24 keys and thus need more data. It is better 
    to just learn 2 keys, C major and A minor.
    
    Arguments:
    song (m21 stream): Piece to transpose
    
    Returns
    transposed_song (m21 stream)
    """

    print("\n\tTransposing song")
    
    # Look at the key
    key = None
    
    # Try to extract it from the object
    try:
        key = song.keySignature
    except:
        pass
    
    # If this object is not a m21.Key object, estimate it
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
            
    print(f"\t\tKey is {key.name}")    
    
    # get interval for transposition, i.e., if key = Bmaj, how much to move from Bmaj to Cmaj?
    if key.mode == "major":    #if the key is major calculate how to move to Cmaj
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":  #if the key is minor calculate how to move to Amin
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song, time_step_duration = TIME_STEP_DURATION):
    """
    Converts a score into a time-series-like music representation. Each item in the encoded list represents
    quarter lengths and is appended by a space character. The symbols used at each step are: 
        - integers for MIDI number notes
        * 60 = C4
        * 61 = C4#
        * 62 = D4
        * 63 = D4#
        * 64 = E4
        * 65 = F4
        * 66 = F4#
        * 67 = G4
        * 68 = G4#
        * 69 = A4
        * 70 = A4#
        * 71 = B4
        * 72 = C5
        - 'r' for representing a rest
        - '_' for representing notes/rests
        - X,Y,Z,W for chords (list of integers connected by commas)
    
    Here's a sample encoding:
    
    C4 note for 4 time steps would be:
        [60 _ _ _]
        Pitch Symbol = 60
            
    Dmaj (D + F# + A) chord for 2 time steps would be:
        [62,66,69 _]
    
    Arguments:
    -song (m21 stream): Piece to encode
    -time_step_duration (float): Duration of each time step in quarter length

    Returns:
        encoded song (list of str)
    """
    
    # Documentation about General Note -> base class for {Note, Rest, Chord}
    # https://web.mit.edu/music21/doc/moduleReference/moduleNote.html#music21.note.GeneralNote
    
    # About Chords
    # https://web.mit.edu/music21/doc/moduleReference/moduleChord.html#music21.chord.Chord
    
    print("\n\tEncoding song...")
    
    encoded_song = []

    # Flattening all the events of the song and consider just notes, chords and rests
    parts = song.getElementsByClass(m21.stream.Part)
    part0_flat = parts[0].flat
    nar0 = part0_flat.notesAndRests
    note_chords_rests_list = song.flat.notesAndRests
    for event in note_chords_rests_list:
        
        symbol = None
        
        # (A) handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        
        # (B) handle chords
        elif isinstance(event, m21.chord.Chord):
            # List of pitches in the chord, i.e., Dmaj would be [62,66,69]
            # Need to get them sorted to avoid duplicates, such as [62,66,69] and [66,62,69]
            sorted_pitches_by_midi = sorted([p.midi for p in event.pitches])
            midi_pitches = [str(i) for i in sorted_pitches_by_midi]
            # Append them in a string object sepparated by comma
            symbol = ",".join(midi_pitches) # [62,66,69] <-> "62,66,69"
        
        # (C) handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        else:
            raise Exception("don't know class of element {event}")
        
        # convert the event into time series notation
        event_duration = event.duration.quarterLength
        steps = int(event_duration/ time_step_duration)
        for step in range(steps):

            # If it's the first time we see a event, let's encode it.
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
    filename_and_songs = load_songs(songs_path) #list of tuples (file_name, m21.Score)
    quantity_songs = len(filename_and_songs)
    max_digits = len(str(quantity_songs))
          
    # Enumerate song one by one, indexing by i
    saved_songs = 0
    for i, filename_song in enumerate(filename_and_songs):

        song_name, song = filename_song # retrieve the info from the tuple     
        print(f"\nAnalyzing song #{i+1} named {song_name}")        
        
        # 1. Check note's duration compatibility
        are_durations_acceptable = has_acceptable_durations(song)
        
        # 2. For now just allow streams with a single part
        is_single_parted = len(song.getElementsByClass(m21.stream.Part)) == 1
        
        if are_durations_acceptable and is_single_parted:
        
            # 3. Transpose the song to Cmaj or Amin key
            # transposed_song = transpose(song)
            
            # 4. Encode each song with music time series representation
            encoded_song = encode_song(song)
    
            # 5. Save encoded song to text file
            encoded_file_name = "encoded song " + str(i).zfill(max_digits) + " " + song_name
            create_plain_file(ENCODED_SONGS_FOLDER_PATH, 
                              encoded_file_name, 
                              encoded_song,
                              "txt")
            print("\t\tencoded song created as text file")
            
            # 6. Sanity check: create the file just encoded as a midi file to check-hear it
            create_midi_file_from_encoded_text_file(encoded_file_name, 
                                                     ENCODED_SONGS_FOLDER_PATH, 
                                                     encoded_file_name,
                                                     TIME_STEP_DURATION)
            print("\t\tencoded song created as midi file as well")
            
            # Update counter
            saved_songs += 1
        
        else:
            print(f"\nSong #{i+1} named {song_name} was not processed")
        
    print(f"\nnumber of encoded songs: {saved_songs}")


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
            file_name, file_extension = os.path.splitext(file)
            
            # Just consider text files
            if file_extension == ".txt":            
                file_path = os.path.join(path, file) 
                song = load_plain_file(file_path) # is a string like "60 _ _ _ 61 _ _ _ 62 _ 66 _ ..."
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
    vocabulary = list(set(songs.split())) #the set eliminate duplicates symbols
    
    # create mappings
    print(f"\tthere are {len(vocabulary)} unique symbols")
    for i, symbol in enumerate(vocabulary):
        mappings_symbol_to_index[symbol] = i
        mappings_index_to_symbol[i] = symbol
        #print(f"\t\t{symbol} <-> {i}")

    # save both dictionaries to a json file
    ordered_dict_symbol_to_index = collections.OrderedDict(sorted(mappings_symbol_to_index.items()))
    with open(MAPPING_SYMBOL_TO_INDEX_PATH, "w") as fp: #open the file in writing mode
        json.dump(ordered_dict_symbol_to_index, fp, indent=4) #indent to easier reading
    
    
    with open(MAPPING_INDEX_TO_SYMBOL_PATH, "w") as fp: #open the file in writing mode
        json.dump(mappings_index_to_symbol, fp, indent=4) #indent to easier reading

    print("\tdictionaries created")

def map_from_symbol_to_indexes(mapped_songs_file):
    """
    Takes a list of symbols as string separated by " " and returns it as a list of integers (keys)
    
    Arguments:
        mapped_songs_file (str): lots of songs appended together as string symbols
    
    Returns:
        song_encoded_as_indexes -> a list of integers with their symbols encoded as integers
    """
    song_encoded_as_indexes = []

    # load mappings
    with open(MAPPING_SYMBOL_TO_INDEX_PATH, "r") as fp: #read mode
        mappings_symbol_to_index = json.load(fp)

    # transform songs string to list of symbols
    list_of_symbols = mapped_songs_file.split()

    # map each element to its index
    for symbol in list_of_symbols:
        song_encoded_as_indexes.append(mappings_symbol_to_index[symbol])

    return song_encoded_as_indexes

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
    encoded_single_file = load_plain_file(SINGLE_FILE_DATASET_PATH)
    song_encoded_as_indexes = map_from_symbol_to_indexes(encoded_single_file)

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
    songs = create_single_file_dataset()
    create_mapping(songs)
    print("\nPreprocessing finished!")

if __name__ == "__main__":
    main()


