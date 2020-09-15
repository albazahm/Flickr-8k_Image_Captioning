import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import pickle

def load_captions(filepath='flickr-8k/captions.txt'):

    """
    Function to load captions and image filenames into dataframe

    Parameters
    ----------
    filepath: string
        The filepath where the textfile is located

    Returns
    -------
    captions_df: Pandas DataFrame
        Dataframe containing captions and image filenames data
    """


    captions_df = pd.read_csv(filepath)
    return captions_df

def remove_punctuation_strip_text(row):

    """
    Function to remove punctuation and strip text from a row of a DataFrame

    Parameters
    ----------
    row: Pandas DataFrame row
        Intended to use with the pandas.DataFrame.apply function on each row of dataframe

    Returns
    -------
    processed_text: list
        list of lists containing strings corresponding to captions
    """

    processed_text = [word.strip() for word in row if not word.isnumeric() and word not in string.punctuation]
    return processed_text

def add_start_end_tokens(row, start_token='*start*', end_token='*end*'):

    """
    Function to add tokens indicating the start and end of a caption

    Parameters
    ----------
    row: Pandas DataFrame row
        Intended to use with the pandas.DataFrame.apply function on each row of dataframe
    start_token: string
        Token to be added that indicates the start of the caption sequence
    end_token: string
        Token to be added that indicates the end of the caption sequence
    
    Returns
    -------
    processed_text: list
        list of lists containing string corresponding to strings with the start_token appended at the beginning
        and the end_token appended at the end
    """
    
    processed_text = [start_token] + row + [end_token]
    return processed_text

def preprocess_text(dataframe):

    """
    Function to add tokens indicating the start and end of a caption

    Parameters
    ----------
    dataframe: Pandas DataFrame
        Dataframe to be preprocessed
    
    Returns
    -------
    text: list
        list of lists of strings corresponding to preprocessed captions
    image_filenames: list
        list of strings corresponding to which each caption corresponds
    unique_words: set
        set of unique words in text
    word_to_ind: dict
        Dictionary containing mappings of words to indicies
    ind_to_word: dict
        Dictionary containing mappings of indicies to words
    """

    dataframe['caption'] = dataframe['caption'].str.lower()
    dataframe['caption'] = dataframe['caption'].str.split(' ')
    dataframe['caption'] = dataframe['caption'].apply(remove_punctuation_strip_text)
    dataframe['caption'] = dataframe['caption'].apply(add_start_end_tokens)
    text = dataframe['caption'].values
    words = [word for item in text for word in item]
    words.sort()
    unique_words = set(words)
    word_to_ind = {word:ind for ind, word in enumerate(unique_words)}
    ind_to_word = {ind:word for word, ind in word_to_ind.items()}
    image_filenames = dataframe['image'].values

    return text, image_filenames, unique_words, word_to_ind, ind_to_word

def words_to_indices(text, mapping):

    
    """
    Function that applies a mapping to text

    Parameters
    ----------
    text: list
        list of lists of strings corresponding to captions
    mapping: dict
        Dictionary contaning mapping of words to indices
    
    Returns
    -------
    processed_text: list
        list of lists containing string corresponding to strings with the start_token appended at the beginning
        and the end_token appended at the end
    """

    text = [list(map(mapping.get, item)) for item in text] 
    return text

def create_sequences(captions, filenames, mapping):

    """
    Function that creates sequences from text

    Parameters
    ----------
    captions: list
        list of lists of strings corresponding to captions
    filenames: list
        list of strings corresponding to image filenames
    mapping: dict
        Dictionary contaninig mapping of words to indices
    
    Returns
    -------
    X: NumPy array
        array of lists containing sequences
    y: list
        list of strings containing target sequence
    filenames: list
        list of strings corresponding to filenames
    """

    #finding the maximum length of a caption for padding
    max_length = np.max([len(item) for item in captions])
    #mapping the word to indices
    indices = words_to_indices(captions, mapping)
    
    #creating lists for filenames, caption sequences and target sequences
    filenames = [item[0] for item in zip(filenames, indices) for i in range(len(item[1])-1)]
    X = [item[:i+1] for item in indices for i in range(len(item)-1)]
    X = pad_sequences(X, maxlen=max_length)
    y = [[item[i+1]] for item in indices for i in range(len(item)-1)]
    
    return X, y, filenames

if __name__ == '__main__':
    
    #load dataframe
    df = load_captions()
    #preprocess dataframe
    captions, filenames, unique_words, word_to_ind, ind_to_word = preprocess_text(df)
    #create sequences from captions
    X, y, filenames = create_sequences(
        captions=captions,
        filenames=filenames,
        mapping=word_to_ind
    )
    #dump sequences, target sequence and filenames as pickle
    pickle.dump(
        (X, y, filenames),
        file=open('text_features.p', "wb"))

    #dump mappings of word-to-ind and in-to-word as pickle
    pickle.dump(
        (word_to_ind, ind_to_word),
        file=open("mappings.p", "wb")
    )

