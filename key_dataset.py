# README
# Phillip Long
# August 10, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library.

# python ./key_dataset.py labels_filepath output_filepath audio_dir


# IMPORTS
##################################################
import sys
from os.path import exists, join, dirname
from os import makedirs, remove
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import torchvision.transforms
import pandas as pd
# sys.argv = ("./key_dataset.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv", "/Volumes/Seagate/artificial_dj_data/key_data")
##################################################


# CONSTANTS
##################################################
SAMPLE_RATE = 44100 // 2
SAMPLE_DURATION = 20.0 # in seconds
STEP_SIZE = SAMPLE_DURATION / 4 # in seconds, the amount of time between the start of each .wav file
TORCHVISION_MIN_IMAGE_DIM = 224 # 224 is the minimum image width for PyTorch image processing, for waveform to melspectrogram transformation
IMAGE_HEIGHT = 256 # height of the resulting image after transforms are applied
N_FFT = min(1024, (2 * SAMPLE_DURATION * SAMPLE_RATE) // TORCHVISION_MIN_IMAGE_DIM) # number of samples in each bin on the mel spectrogram
N_MELS = IMAGE_HEIGHT // 2 # for waveform to melspectrogram transformation
SET_TYPES = {"train": 0.7, "validate": 0.2, "test": 0.1} # train-validation-test fractions
# KEY, KEY_CLASS, and KEY_QUALITY mappings in circle of fifths order
KEY_MAPPINGS = ("C", "Am",
                "G", "Em",
                "D", "Bm",
                "A", "F#m",
                "E", "C#m",
                "B", "G#m",
                "F#", "D#m",
                "C#", "A#m",
                "G#", "Fm",
                "D#", "Cm",
                "A#", "Gm",
                "F", "Dm")
KEY_CLASS_MAPPINGS = tuple(f"{KEY_MAPPINGS[i]} / {KEY_MAPPINGS[i + 1]}" for i in range(0, len(KEY_MAPPINGS), 2)) # ("C / Am", ... , "F / Dm")
KEY_QUALITY_MAPPINGS = tuple(('Major', 'minor'))
##################################################


# KEY DATASET OBJECT CLASS
##################################################

class key_dataset(Dataset):

    def __init__(self, labels_filepath, set_type, device, use_pseudo_replicates = True):
        # set_type can take on one of three values: ("train", "validate", "test")

        # import labelled data file, preprocess
        # it is assumed that the data are mono wav files
        self.data = pd.read_csv(labels_filepath, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
        self.data = self.data[self.data["path"].apply(lambda path: exists(path))] # remove files that do not exist
        self.data = self.data[~pd.isna(self.data["key"])] # remove na values
        if not use_pseudo_replicates: # if no pseudo-replicates, transform self.data once more
            self.data = self.data.groupby(["title", "artist", "tempo", "path_origin"]).sample(n = 1, replace = False, random_state = 0, ignore_index = True) # randomly pick a sample from each song
            self.data = self.data.reset_index(drop = True) # reset indicies
        
        # partition into the train, validation, or test dataset
        self.data = self.data.sample(frac = 1, replace = False, random_state = 0, ignore_index = True) # shuffle data
        set_type = "" if set_type.lower()[:3] not in map(lambda key: key[:3], SET_TYPES.keys()) else str(*(key for key in SET_TYPES.keys() if key[:3] == set_type.lower()[:3])) # get the SET_TYPES key closest to set_type
        self.data = self.data.iloc[_partition(n = len(self.data))[set_type]].reset_index(drop = True) # extract range depending on set_type, also reset indicies
        
        # import constants
        self.device = device

        # instantiate transformation functions
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = N_FFT, n_mels = N_MELS).to(self.device) # make sure to adjust MelSpectrogram parameters such that # of mels > 224 and ceil((2 * SAMPLE_DURATION * SAMPLE_RATE) / (n_fft)) > 224
        self.normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).to(self.device) # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # returns the transformed signal and the actual key value
        return self._get_signal(index = index), torch.tensor([self.data.at[index, "key"]], dtype = torch.uint8)
    
    # get info (title, artist, original filepath) of a file given its index; return as dictionary
    def get_info(self, index):
        info = self.data.loc[index, ["title", "artist", "path_origin", "path", "tempo", "key", "key_class", "key_quality"]].to_dict()
        info["key_name"] = get_key_name(index = int(info["key"]))
        info["key_class_name"] = get_key_class_name(index = int(info["key_class"]))
        info["key_quality_name"] = get_key_quality_name(index = int(info["key_quality"]))
        return info
    
    # get the audio signal as a torch Tensor and apply transformations
    def _get_signal(self, index):
        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(self.data.at[index, "path"], format = "wav") # returns the waveform data and sample rate
        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(self.device)
        # apply transformations
        signal = self._transform(signal = signal, sample_rate = sample_rate)
        # return the signal as a transformed tensor registered to the correct device
        return signal

    # transform a waveform into whatever will be used to train a neural network
    def _transform(self, signal, sample_rate):

        # resample; sample_rate was already set in preprocessing
        # signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = self.device) # resample for consistent sample rate
        
        # convert from stereo to mono; already done in preprocessing
        # signal = _mix_down_if_necessary(signal = signal)

        # pad/crop for fixed signal duration; duration was already set in preprocessing
        # signal = _edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = SAMPLE_DURATION) # crop/pad if signal is too long/short

        # convert waveform to melspectrogram
        signal = self.mel_spectrogram(signal) # (single channel, # of mels, # of time samples) = (1, 128, ceil((SAMPLE_DURATION * SAMPLE_RATE) / (n_fft = 1024)) = 431)

        # perform local min-max normalization such that the pixel values span from 0 to 255 (inclusive)
        signal = (signal - torch.min(signal).item()) * (255 / (torch.max(signal).item() - torch.min(signal).item()))

        # make image height satisfy PyTorch image processing requirements, (1, 128, 431) -> (1, 256, 431)
        signal = torch.repeat_interleave(input = signal, repeats = IMAGE_HEIGHT // N_MELS, dim = 1)

        # convert from 1 channel to 3 channels (mono -> RGB); I will treat this as an image classification problem
        signal = torch.repeat_interleave(input = signal, repeats = 3, dim = 0) # (3 channels, # of mels, # of time samples) = (3, 256, 431)

        # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)
        signal = self.normalize(signal)

        return signal
    
    # sample n_predictions random rows from data, return a tensor of the audios and a tensor of the labels
    # def sample(self, n_predictions):
    #     inputs_targets = [self.__getitem__(index = i) for i in self.data.sample(n = n_predictions, replace = False, ignore_index = False).index]
    #     inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0).to(self.device) # key_nn expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze()
    #     targets = torch.cat([input_target[1] for input_target in inputs_targets], dim = 0).view(n_predictions, 1).to(self.device) # note that I register the inputs and targets tensors to whatever device we are using
    #     del inputs_targets
    #     return inputs, targets

##################################################

# KEY_CLASS AND KEY_QUALITY CHILD CLASSES
##################################################

# key class (return one of 12 relative keys)
class key_class_dataset(key_dataset):

    def __init__(self, labels_filepath, set_type, device):
        super().__init__(labels_filepath, set_type, device)

    def __getitem__(self, index):
        # returns the transformed signal and the actual key class value
        return super()._get_signal(index = index), torch.tensor([self.data.at[index, "key_class"]], dtype = torch.int64)


# key quality (return Maj or min)
class key_quality_dataset(key_dataset):

    def __init__(self, labels_filepath, set_type, device):
        super().__init__(labels_filepath, set_type, device)

    def __getitem__(self, index):
        # returns the transformed signal and the actual key quality value
        return super()._get_signal(index = index), torch.tensor([self.data.at[index, "key_quality"]], dtype = torch.float32) # 0 for Major, 1 for minor

##################################################


# HELPER FUNCTIONS
##################################################

# partition dataset into training, validation, and test sets
def _partition(n, set_types = SET_TYPES):
    set_types_values = [int(i.item()) for i in (torch.cumsum(input = torch.Tensor([0,] + list(set_types.values())), dim = 0) * n)] # get indicies for start of each new dataset type
    set_types_values = [range(set_types_values[i - 1], set_types_values[i]) for i in range(1, len(set_types_values))] # create ranges from previously created indicies
    set_types = dict(zip(set_types.keys(), set_types_values)) # create new set types dictionary
    set_types[""] = range(n) # create instance where no set type is named, so return all values
    return set_types

# resampler
def _resample_if_necessary(signal, sample_rate, new_sample_rate, device):
        if sample_rate != new_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = new_sample_rate).to(device)
            signal = resampler(signal)
        return signal, new_sample_rate

# convert from stereo to mono
def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1: # signal.shape[0] = # of channels; if # of channels is more than one, it is stereo, so convert to mono
        signal = torch.mean(signal, dim = 0, keepdim = True)
    return signal

# crop/pad if waveform is too long/short
def _edit_duration_if_necessary(signal, sample_rate, target_duration):
    n = int(target_duration * sample_rate) # n = desired signal length in # of samples; convert from seconds to # of samples
    if signal.shape[1] > n: # crop if too long
        signal = signal[:, :n]
    elif signal.shape[1] < n: # zero pad if too short
        last_dim_padding = (0, n - signal.shape[1])
        signal = torch.nn.functional.pad(signal, pad = last_dim_padding, value = 0)
    return signal

# given a mono signal, return the sample #s for which the song begins and ends (trimming silence)
def _trim_silence(signal, sample_rate, window_size = 0.1): # window_size = size of rolling window (in SECONDS)
    # preprocess signal
    signal = torch.flatten(input = signal) # flatten signal into 1D tensor
    signal = torch.abs(input = signal) # make all values positive
    # parse signal with a sliding window
    window_size = int(sample_rate * window_size) # convert window size from seconds to # of samples
    starting_frames = tuple(range(0, len(signal), window_size)) # determine starting frames
    is_silence = [torch.mean(input = signal[i:(i + window_size)]).item() for i in starting_frames] # slide window over audio and get average level for each window
    # determine a threshold to cutoff audio
    threshold = max(is_silence) * 1e-4 # determine a threshold, ADJUST THIS VALUE TO ADJUST THE CUTOFF THRESHOLD
    is_silence = [x < threshold for x in is_silence] # determine which windows are below the threshold
    start_frame = starting_frames[is_silence.index(False)] if sum(is_silence) != len(is_silence) else 0 # get starting frame of audible audio
    end_frame = starting_frames[len(is_silence) - is_silence[::-1].index(False) - 1] if sum(is_silence) != len(is_silence) else 0 # get ending from of audible audio
    return start_frame, end_frame

##################################################


# ACCESSOR METHODS
##################################################

# KEY
# get the key index from key name
def get_key_index(name):
    return KEY_MAPPINGS.index(name)

# get the key name from the key index
def get_key_name(index):
    return KEY_MAPPINGS[index]

# get the key from the key class index and key quality index
def get_key(key_class_index, key_quality_index):
    return KEY_MAPPINGS[(2 * key_class_index) + key_quality_index]

# KEY CLASS
# get the key class index from key class name
def get_key_class_index(name):
    key_classes = [i for i in range(len(KEY_CLASS_MAPPINGS)) if name.lower() in KEY_CLASS_MAPPINGS[i].lower()]
    return int(sum(key_classes))

# get the key class name from the key class index
def get_key_class_name(index):
    return KEY_CLASS_MAPPINGS[index]


# KEY QUALITY
# get the key quality index from key quality name
def get_key_quality_index(name):
    key_quality = 0 if KEY_QUALITY_MAPPINGS[0] in name else 1
    return key_quality

# get the key quality name from the key quality index
def get_key_quality_name(index):
    return KEY_QUALITY_MAPPINGS[index]

##################################################


# if __name__ == "__main__" only runs the code inside the if statement when the program is run directly by the Python interpreter.
# The code inside the if statement is not executed when the file's code is imported as a module.
if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    AUDIO_DIR = sys.argv[3]
    ##################################################


    # CREATE AND PREPROCESS WAV FILE CHOPS FROM FULL SONGS
    ##################################################

    # create audio output directory and output_filepath
    if not exists(AUDIO_DIR): 
        makedirs(AUDIO_DIR)
    if not exists(dirname(OUTPUT_FILEPATH)):
        makedirs(dirname(OUTPUT_FILEPATH))   

    # clear AUDIO_DIR
    for filepath in tqdm(glob(join(AUDIO_DIR, "*")), desc = f"Clearing files from {AUDIO_DIR}"):
        remove(filepath)
    
    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # load in labels
    data = pd.read_csv(LABELS_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    data = data[data["path"].apply(lambda path: exists(path))] # remove files that do not exist
    data = data[~pd.isna(data["key"])] # remove NA and unclassified keys
    data = data.reset_index(drop = True) # reset indicies
    
    # loop through songs and create .wav files
    origin_filepaths, output_filepaths, keys, key_classes, key_qualities = [], [], [], [], []
    for i in tqdm(data.index, desc = "Chopping up songs into WAV files"): # start from start index

        # preprocess audio
        try: # try to import the file
            signal, sample_rate = torchaudio.load(data.at[i, "path"], format = "mp3") # load in the audio file
        except RuntimeError:
            continue
        signal = signal.to(device) # register signal onto device (gpu [cuda] or cpu)
        signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = device) # resample for consistent sample rate
        signal = _mix_down_if_necessary(signal = signal) # if there are multiple channels, convert from stereo to mono

        # chop audio into many wav files
        start_frame, end_frame = _trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts
        origin_filepath = data.at[i, "path"] # set original filepath
        key_index = get_key_index(name = data.at[i, "key"]) # set key
        key_class_index = get_key_class_index(name = data.at[i, "key"]) # set key class
        key_quality_index = get_key_quality_index(name = data.at[i, "key"]) # set key quality
        for j, starting_frame in enumerate(starting_frames):
            output_filepath = join(AUDIO_DIR, f"{i}_{j}.wav") # create filepath
            torchaudio.save(output_filepath, signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate, format = "wav") # save chop as .wav file
            origin_filepaths.append(origin_filepath) # add original filepath to origin_filepaths
            output_filepaths.append(output_filepath) # add filepath to output_filepaths
            keys.append(key_index) # add key to keys
            key_classes.append(key_class_index) # add key class to key classes
            key_qualities.append(key_quality_index) # add key quality to key qualities
        
    # write to OUTPUT_FILEPATH
    data = data.rename(columns = {"path": "path_origin"}).drop(columns = ["key"]) # rename path column in the original dataframe
    key_data = pd.DataFrame(data = {"path_origin": origin_filepaths, "path": output_filepaths, "key": keys, "key_class": key_classes, "key_quality": key_qualities}) # create key_data dataframe
    key_data = pd.merge(key_data, data, on = "path_origin", how = "left").reset_index(drop = True) # left-join key_data and data
    key_data = key_data[["title", "artist", "tempo", "path_origin", "path", "key", "key_class", "key_quality"]] # select columns
    # most of the information in key_data is merely to help me locate a file if it causes problem; in an ideal world, I should be able to ignore it
    print(f"\nWriting output to {OUTPUT_FILEPATH}.")
    key_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA") # write output

    ##################################################


    # TEST DATASET OBJECT
    ##################################################

    # instantiate key dataset
    key_data = key_dataset(labels_filepath = OUTPUT_FILEPATH, set_type = "", device = device)

    # test len() functionality
    print(f"There are {len(key_data)} samples in the dataset.")

    # test __getitem__ functionality
    signal, label = key_data[0]

    # test get_info() functionality
    print(f"The artist of the 0th sample is {key_data.get_info(0)['artist']}.")

    ##################################################


    # CODE FOR ADDING KEY CLASS AND KEY QUALITY IN HINDSIGHT
    ##################################################

    # key_data = pd.read_csv(OUTPUT_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    # key_names = key_data["key"].apply(get_key_name)
    # key_data["key_class"] = key_names.apply(get_key_class_index)
    # key_data["key_quality"] = key_names.apply(get_key_quality_index)
    # key_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA") # write output

    ##################################################
