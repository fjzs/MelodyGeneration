# Adapted from Valerio Velardo, The Sound of AI
# https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
# https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

import numpy as np
import keras as K
from preprocess import DIRECTORY, ENCODED_SONGS_FOLDER_PATH
from train import MODEL_FILE, SEQUENCE_LENGTH_TX
from midi_index_utils import VOCABULARY_SIZE, idxsToMidi
from file_utils import load_plain_file, save_midi_file_from_midi_object



# ------------------------- Music creation parameters -----------------------------#
CREATIONS_DIRECTORY = DIRECTORY + "/3. creations"

# For each creation, specify the characteristics in lists
TEMPERATURES = [1] 
CREATION_LENGTH_STEPS = [10000, 100, 100]
SEED_BASE_FILE = ["encoded song 0004 MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--1.txt",
                  "encoded song 0004 MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--1.txt",
                  "encoded song 0004 MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--1.txt"]
SEED_BASE_FILE_FIRST_ELEMENTS = [SEQUENCE_LENGTH_TX, 
                                 SEQUENCE_LENGTH_TX, 
                                 SEQUENCE_LENGTH_TX]
    
# ---------------------------------------------------------------------------------#


class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, 
                 temperature: float, 
                 creation_length_steps: int, 
                 seed_base_file: str, 
                 seed_base_file_first_elements: int, 
                 model_path: str = MODEL_FILE):
        """
        Constructor that initialises a TensorFlow model

        Args:
            temperature (float): controls how much randomness is involved
            creation_length_steps (int): how many steps to be synthesized
            seed_base_file (str): what file to use as a seed
            seed_base_file_first_elements (int): how many steps to use as a seed
            model_path (str, optional): Defaults to MODEL_FILE.

        Returns:
            None.
        """       

        # Instance attributes
        self.__temperature = temperature
        self.__creation_length_steps = creation_length_steps
        self.__seed_base_file = seed_base_file
        self.__seed_base_file_first_elements = seed_base_file_first_elements
        self.__model = K.models.load_model(model_path)

    def generate_melody(self):
        """
        Generates a melody using the model and returns a list of int as the encoded events
        
        Returns:
        - melody_list_indices (list of int): List with indices
        """
        
        # Initialize output
        melody_list_indices = []
        
        # Get the seed indices
        seed_file_path = ENCODED_SONGS_FOLDER_PATH + "/" + self.__seed_base_file
        encoded_song = load_plain_file(seed_file_path)
        indices_list_str = encoded_song.split(',') #these are string elements
        for i,symbol in enumerate(indices_list_str):  
            if i <= self.__seed_base_file_first_elements:
                melody_list_indices.append(int(symbol))
            else:
                break
                

        # Synthetize each step t of all the num_steps to be generated
        print(f"Starting synthetizing song with {self.__creation_length_steps} steps")
        for t in range(self.__creation_length_steps):

            # Update the current seed as the last Tx elementss
            current_seed = melody_list_indices[-SEQUENCE_LENGTH_TX:]

            # One-hot encode the current seed with shape (m, Tx, num of symbols)
            onehot_seed = K.utils.to_categorical(current_seed, num_classes = VOCABULARY_SIZE)            
            onehot_seed = onehot_seed[np.newaxis, ...] # add the first dimension as requested by Keras

            # Predicting
            # probabilities has shape (m, vocabulary size), given that we have just one sample, we index it by zero
            probabilities = self.__model.predict(onehot_seed)[0] # Example [0.1, 0.2, 0.1, 0.6]
            sampled_index = self.sample(probabilities)

            # Update the creation
            melody_list_indices.append(sampled_index)
            print(f"Index @ t={t} is {sampled_index}")

        return melody_list_indices


    def sample(self, probabilites):
        """
        Samples an index from a probability array reapplying softmax using temperature.
        It is somewhat similiar to the simmulated annealing metaheuristic method
        - If t° -> infinity => we are in a hot room, stochastic environment, probability distribution flattens
        - If t° = 0         => we are in a cold room, deterministic environment, single pick the highest probability
        
        Arguments:
        - probabilites (nd.array): Array containing probabilities for each of the possible outputs.
                
        Returns:
        - index (int): Selected output symbol
        """
        
        index = None
        
        # deterministic enviroment
        if self.__temperature == 0:
            index = np.argmax(probabilites)
        
        # stochastic environment
        else:
            predictions = np.log(probabilites) / self.__temperature
            probabilites = np.exp(predictions) / np.sum(np.exp(predictions)) #softmax function applied
            choices = range(len(probabilites))
            index = np.random.choice(choices, p = probabilites)        

        return index   


if __name__ == "__main__":
    index = 0
    t = TEMPERATURES[index]
    creation_steps = CREATION_LENGTH_STEPS[index]
    seed_base_file = SEED_BASE_FILE[index]
    first_elements = SEED_BASE_FILE_FIRST_ELEMENTS[index]
    
    mg = MelodyGenerator(t, creation_steps, seed_base_file, first_elements)    
    melody_indices = mg.generate_melody()
    midi_file, errors = idxsToMidi(melody_indices, verbose = True)
    save_midi_file_from_midi_object(midi_file, CREATIONS_DIRECTORY, "angy_cambios.mid")
    
    