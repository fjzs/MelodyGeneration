import os
import music21 as m21



def create_midi_file_from_stream(song, directory, file_name):
    """
    Creates a midi file in a defined directory

    Args:
        song (Music21 Stream): the song to be created as .mid
        directory (str): desired directory
        file_name (str): file name + extension, such as "song.mid"

    Returns:
        None.

    """
    
    # Create folder if not existant
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create the file
    fullpath = directory + "/" + file_name
    song.write('midi', fp = fullpath)

def create_plain_file(directory, file_name, content, extension):
    
    # Create folder if not existant
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = directory + "/" + file_name + "." + extension
    
    # Write
    with open(file_path, "w") as fp:
        fp.write(content)

def load_plain_file(file_path):
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

def create_general_note(quarter_length_duration, symbol):
    """
    Creates a GeneralNote instance for a given symbol and duration. GeneralNote is a 
    base class for a Chord, Note or Rest

    Args:
        quarter_length_duration (float): duration in quarter lengths.
        symbol (str): string symbol of the notes, can be {X, "r", [X,Y,Z,...]}, where X, Y, Z are integer numbers

    Returns:
        general_note (GeneralNote): the GeneralNote object.

    """
        
    #https://web.mit.edu/music21/doc/moduleReference/moduleNote.html#music21.note.GeneralNote
    general_note = None
    
    # separate the symbols in case a chord is being sent
    symbol_list = symbol.split(',')
    
    # handle a rest
    if symbol_list[0] == "r": 
        general_note = m21.note.Rest(quarterLength = quarter_length_duration)
    
    # handle notes and chords
    else:
        midi_pitches = [int(s) for s in symbol_list]
        
        # handle a single note
        if len(midi_pitches) == 1:
            general_note = m21.note.Note(midi_pitches[0], quarterLength = quarter_length_duration)
        
        # handle chords
        else:
            general_note = m21.chord.Chord(midi_pitches, quarterLength = quarter_length_duration)
    
    return general_note

def create_midi_file_from_encoded_text_file(plain_file_name, directory, new_file_name, time_step_duration, extension = "mid"):
    """
    Creates a midi file from a encoded text file.
    Useful to pre-check the midi representation of the encoded files we are passing to the model.

    Args:
        plain_file_name (str): self-explainable
        new_file_name (str): self-explainable
        directory (str): self-explainable
        time_step_duration (float): in quarter lengths, ie, if time_step_duration = 0.5 => 1 quarter length = 2 time steps
        extension (str): self-explainable, defaults to "mid"
    Returns:
        None.
    """
    
    # create a music21 stream to append the events (note and rests)
    stream = m21.stream.Stream()

    """
    parse all the symbols in the melody and create note/rest objects
    example: 60 _ _ _ 62 _ r _ _ would be:
        C4    time steps = 4
        D4    time steps = 2
        Rest  time steps = 3
    
    To to this we have 3 auxiliar variables:
        current_event is to keep track of the event (note or rest) about to be inserted
        step_counter tracks the duration of the current_event
        non_event_symbols to detect event symbols (complement)
    """
    
    symbol_list = load_plain_file(directory + "/" + plain_file_name + ".txt").split()
    
    # Initialize auxiliar variables
    current_event = None
    time_step_counter = 0
    non_event_symbols = ["_", "/"]
    for t in range(len(symbol_list)):
        
        try:
            
            # Look at the symbol at time = t
            symbol_t = symbol_list[t]
            
            # If symbol_t is an event, replace the existing current_event for this one and start the counter
            if symbol_t not in non_event_symbols:
                current_event = symbol_list[t]
                time_step_counter = 1
                
            # If we are in a prolonged event "_", increase the step_counter by 1
            elif symbol_t == "_":
                time_step_counter +=1
            
            # Event is finishing? 
            # i.e., the next event starts or the melody ends --> write it to the stream
            if ((t+1 == len(symbol_list)) or (symbol_list[t+1] not in non_event_symbols)):
                
                # Calculate the quarter length duration of the event
                quarter_length_duration = time_step_counter * time_step_duration
                
                # Create the event
                m21_event = create_general_note(quarter_length_duration, current_event)                
                
                # write the event to the stream
                stream.append(m21_event) 
        except:
            raise Exception(f"Error when generating note @ t = {t}")
    
    # Create folder if not existant
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # write the m21 stream to a midi file
    name_plus_extension = new_file_name + "." + extension
    path = directory + "/" + name_plus_extension    
    stream.write(fmt = "midi", fp = path)
    
    print("\t\tencoded song created as midi file (sanity check)")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
