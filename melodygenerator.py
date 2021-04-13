# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import json
import numpy as np
import keras as K
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_SYMBOL_TO_INDEX_PATH, MAPPING_INDEX_TO_SYMBOL_PATH, DIRECTORY
from train import SAVE_MODEL_PATH

# ------------------------- Music creation parameters -----------------------------#
TEMPERATURE = 1 # the higher the temperature, more randomness in the creation
MIN_STEPS = 50 # minimum number of steps to be synthetized
MAX_STEPS = 100 # maximum number of steps to be synthetized
SEED = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"    
# ---------------------------------------------------------------------------------#



class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path = SAVE_MODEL_PATH):
        """
        Constructor that initialises TensorFlow model
        
        Arguments:
        - model_path (str): the path of the model
        
        Returns:
        - instance of MelodyGenerator
        """

        # Instance attributes
        self.model = K.models.load_model(model_path)
        self.mappings_symbol_to_index = {}
        self.mappings_index_to_symbol = {}
        self.vocabulary_length = 0
        self.start_symbols = ["/"] * 1 # SEQUENCE_LENGTH #TODO FIX THIS
        
        # Read the vocabulary symbol -> index
        with open(MAPPING_SYMBOL_TO_INDEX_PATH, "r") as fp: 
            self.mappings_symbol_to_index = json.load(fp)
            self.vocabulary_length = len(self.mappings_symbol_to_index)
            
        # Read the vocabulary index -> symbol 
        # https://docs.python.org/3/library/json.html given that the key is a str, we format it as int
        with open(MAPPING_INDEX_TO_SYMBOL_PATH, "r") as fp: 
            temp_dict_str_to_symbol = json.load(fp) 
            for key, value in temp_dict_str_to_symbol.items():
                self.mappings_index_to_symbol[int(key)] = value


    def generate_melody(self, 
                        seed_str = SEED, 
                        min_steps = MIN_STEPS, 
                        max_steps = MAX_STEPS,
                        max_sequence_length = SEQUENCE_LENGTH, 
                        temperature = TEMPERATURE):
        """
        Generates a melody using the DL model and returns a midi file.
        
        Arguments:
        - seed_str (str): Melody seed with the notation used to encode the dataset, such as 
        "67 _ _ _ _ _ 65 _ 64,66 _ 62 _ 60 _ _ r"
        - min_steps (int): Minimum number of steps to be generated
        - max_steps (int): Maximum number of steps to be generated
        - max_sequence_len (int): Max number of steps in seed to be considered for generation
        - temperature (float): Numbers closer to 0 make the model more deterministic.
            A higher number makes the generation more unpredictable.
        
        Returns:
        - melody_list_symbols (list of str): List with symbols representing a melody
        """
        
        # Create seed with start symbols
        mapped_start_symbols_to_indexes = [self.mappings_symbol_to_index[s] for s in self.start_symbols]
        mapped_seed_to_indexes = [self.mappings_symbol_to_index[s] for s in seed_str.split()]
                
        # Initialize the seed
        seed_as_list_indexes = mapped_start_symbols_to_indexes.copy() + mapped_seed_to_indexes.copy()
        
        # Initialize the melody
        melody_list_symbols = [] + seed_str.split().copy()

        # Synthetize each step t of all the num_steps to be generated
        print("Starting synthetizing song...")
        for t in range(max_steps):

            # limit the seed to the last max_sequence_length time steps
            seed_as_list_indexes = seed_as_list_indexes[-max_sequence_length:]

            # one-hot encode the seed with shape (m, max_sequence_length, num of symbols)
            onehot_seed = K.utils.to_categorical(seed_as_list_indexes, num_classes = self.vocabulary_length )            
            onehot_seed = onehot_seed[np.newaxis, ...] # add the first dimension as requested by Keras

            # make a prediction
            # probabilities has shape (m, num of symbols), given that we have just one sample, we index it by zero
            probabilities = self.model.predict(onehot_seed)[0] # Example [0.1, 0.2, 0.1, 0.6]
            sampled_index = self.sample_with_temperature(probabilities, temperature)

            # update seed
            seed_as_list_indexes.append(str(sampled_index))

            # map index to symbol
            sampled_symbol = self.mappings_index_to_symbol[sampled_index]
            print(f"\tsampled symbol for t={t} is {sampled_symbol}")
            
            # check whether we're at the end of a melody
            if t >= min_steps and sampled_symbol == "/":
                break
            
            # update melody
            melody_list_symbols.append(sampled_symbol)            

        return melody_list_symbols


    def sample_with_temperature(self, probabilites, temperature):
        """
        Samples an index from a probability array reapplying softmax using temperature.
        It is somewhat similiar to the simmulated annealing metaheuristic method
        - If t° -> infinity => we are in a hot room, stochastic environment, probability distribution flattens
        - If t° -> 0 => we are in a cold room, deterministic environment, single pick the highest probability
        
        Arguments:
        - predictions (nd.array): Array containing probabilities for each of the possible outputs.
        - temperature (float): Float in interval (0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        
        Returns:
        - index (int): Selected output symbol
        """
        
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions)) #softmax function applied
        choices = range(len(probabilites))
        index = np.random.choice(choices, p = probabilites)

        return index

    
    def create_general_note(self, quarter_length_duration, symbol):
        """
        Creates a GeneralNote instance for a given symbol and duration. GeneralNote is a 
        base class for a Chord, Note or Rest

        Args:
            quarter_length_duration (float): duration in quarter lengths.
            symbol (str): string symbol of the notes, can be {X, "r", [X,Y,Z,...]}, where X, Y, Z are integer numbers

        Returns:
            general_note (TYPE): DESCRIPTION.

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
        

    def save_melody(self, melody, temperature = TEMPERATURE, step_duration=0.25, file_extension = "mid"):
        """
        Converts a melody into a MIDI file

        Args:
            - melody (list of str): List of symbols according to the vocabulary
            - temperature (float, optional): Variable for controlling the randomness of the creation. Defaults to TEMPERATURE.
            - step_duration (float, optional): quarter length duration of each step. Defaults to 0.25.
            - file_extension (str, optional): Defaults to "mid".

        Returns:
            None.

        """
        
        # create a music21 stream to append the events (note and rests)
        stream = m21.stream.Stream()

        """
        parse all the symbols in the melody and create note/rest objects
        example: 60 _ _ _ 62 _ r _ _ would be:
            C4    duration = 1/16
            D4    duration = 2/16 
            Rest  duration = 3/16
        
        To to this we have 3 auxiliar variables:
            current_event is to keep track of the event (note or rest) about to be inserted
            step_counter tracks the duration of the current_event
            non_event_symbols to detect event symbols (complement)
        """
        
        # Initialize auxiliar variables
        current_event = None
        step_counter = 0
        non_event_symbols = ["_", "/"]
        for t in range(len(melody)):

            # Look at the symbol at time = t
            symbol_t = melody[t]
            
            # If symbol_t is an event, replace the existing current_event for this one and start the counter
            if symbol_t not in non_event_symbols:
                current_event = melody[t]
                step_counter =1
                
            # If we are in a prolonged event "_", increase the step_counter by 1
            if symbol_t == "_":
                step_counter +=1
            
            # Event is finishing? (the next event starts or the melody ends) then write it to the stream
            if ((t+1 == len(melody)) or (melody[t+1] not in non_event_symbols)):
                
                # Calculate the duration of the event
                quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1
                
                # m21_event = None
                # # handle a rest
                # if current_event == "r":
                #     m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                # # handle a note
                # else:
                #     m21_event = m21.note.Note(int(current_event), quarterLength = quarter_length_duration)
                
                m21_event = self.create_general_note(quarter_length_duration, current_event)                
                
                # write the event to the stream
                stream.append(m21_event)           

        # write the m21 stream to a midi file
        file_name = DIRECTORY + "/len_" + str(len(melody)) + "_temp_"+str(temperature)+"."+file_extension
        stream.write(fmt = "midi", fp = file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    melody = mg.generate_melody()
    print(melody)
    mg.save_melody(melody)