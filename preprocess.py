# Author: Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import os
import music21 as m21

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

KERN_DATASET_PATH = "data/test"

def load_songs_in_kern(dataset_path):
    """
    Loads all kern pieces in dataset using music21.
    
    -param dataset_path (str): Path to dataset
    -return songs (list of m21 streams): List containing all pieces
    """
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        print(f"there are {len(files)} files")
        for file in files:
            # consider only kern files, with extension .krn
            if file[-3:] == "krn": #look at the extension
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song) #Music21 Score                
    return songs

def has_acceptable_durations(song, acceptable_durations):
    """
    Returns True if piece has all acceptable duration, False otherwise.
    
    -param song (m21 stream)
    -param acceptable_durations (list): List of acceptable duration in quarter length
    -return (bool)
    """
    #Looks at all of the notes M21 Objects
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

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

    # get interval for transposition, i.e., if key = Bmaj, how much to move from Bmaj to Cmaj?
    if key.mode == "major":    #if the key is major calculate how to move to Cmaj
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":  #if the key is minor calculate how to move to Amin
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def preprocess(dataset_path):
    """
    1. Loads the songs
    2. Eliminate songs with non acceptable durations
    3. Transpose the to Cmaj/Amin
    4. Encode each song with music time series representation
    5. Save songs to text file
    
    :param dataset_path: path to the files
    :type dataset_path: .krn files
    
    """
        
    # 1. Loads the songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
       
    for song in songs:

        # 2. Eliminate songs with non acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue #skip the song

        # 3. Transpose the to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation

        # save songs to text file

songs = preprocess(KERN_DATASET_PATH)
songs[0].parts[0].show()



