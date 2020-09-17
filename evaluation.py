import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
import pickle
import random
# from project directory
import deep_learning_model
import preprocess_images
import preprocess_text

def create_plot(sample, captions=None, title=None, savename=None):

    """
    Function to nxn grid plot of images and their correspoding captions. If 
    captions are not provided it will assume that the images are part of the
    Flickr-8k dataset.

    Parameters
    ----------
    sample: list
        list of paths to images to be plotted
    captions: list or array, default=None
        list of captions corresponding to the images
    title: string, default=None
        The title of the figure
    savename: string, default=None
        filename to save the resulting plot
    """
    data = pd.read_csv('./flickr-8k/captions.txt')
    dim = int(np.ceil(np.sqrt(len(sample))))
    fig = plt.figure(figsize=(10, 10))
    if title!=None:
        plt.suptitle(title, fontsize=32)
    plt.subplots_adjust(wspace=1, hspace=0.5)
    for ind, i in enumerate(sample):
        if captions==None:
            data_sample = data[data['image']==i.split('/')[-1]]['caption'].values
            label = ('\n'.join(data_sample))
        else:
            data_sample = captions[ind]
            label = (' '.join(data_sample))
        img = image.load_img(i)
        sub_fig = plt.subplot(dim, dim, ind+1)
        plt.imshow(img)
        plt.xlabel(label)

    if savename!=None:
        plt.savefig(savename)

def make_prediction(image_ids, model, word_to_ind, ind_to_word):

    """
    Function to make a caption prediction for a list of image ids

    Parameters
    ----------
    image_ids: list
        list of path to images
    model: Keras Model
        trained model to make the caption prediction
    word_to_ind: dict
        mapping of word to index
    ind_to_word: dict
        mapping of index to word

    Returns
    -------
    predictions: list
        List of lists corresponding to captions for each image

    """
    predictions = []
    vgg_model = preprocess_images.load_prebuilt_model()
    maxlen = model.input_shape[1][1]
    for ii in image_ids:
        prediction = []
        print('Predicting image: ' + str(ii))
        example = image.load_img(ii, target_size=(224, 224))
        example_arr = image.img_to_array(example, dtype='float32')
        example_arr = np.expand_dims(example_arr, axis=0)
        example_arr = preprocess_input(example_arr)
        example_features = vgg_model.predict(example_arr)
        example_features = np.array(example_features).reshape(-1, 4096)
        
        start_string = ['*start*']
        start_ind = list(map(word_to_ind.get, start_string))
        for i in range(maxlen):
            start_seq = pad_sequences([start_ind], maxlen)
            yhat = model.predict([example_features, start_seq])
            yhat = np.argmax(yhat)
            if ind_to_word[yhat] == '*end*':
                break
            prediction.append(ind_to_word[yhat])
            start_ind.append(yhat)
        predictions.append(prediction)
    return predictions