import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Model
from os import listdir
import pickle

def load_preprocess_images(directory='flickr-8k/Images'):

    """
    Function to load and preprocess images from directory. Images are resized to (224, 224),
    transformed to an array and pixels normalized to 0-1 range. 

    Parameters
    ----------
    directory: string
        The directory where the Flikr-8k images dataset is stored
    
    Returns
    -------
    image_dict: dictionary
        A dictionary with filenames as keys and the image arrays as values
    """

    image_dict = {}
    count = 0
    #loop over files in file directory
    for filename in listdir(directory):

        #loading images
        picture = image.load_img(
            path=directory + '/' + filename,
            target_size=(224, 224)
        )
        #transforming images to arrays
        picture_array = image.img_to_array(picture, dtype='float32')

        #adding extra dimension for compatibility with neural network
        picture_array = np.expand_dims(
            picture_array,
            axis=0
            )

        picture_array = preprocess_input(picture_array)
        #storing filename and image array in dictionary
        image_dict[filename] = picture_array
        count += 1
        if count%500==0:
            print(f'{count} images loaded...')
    
    print("done")
    return image_dict

def load_prebuilt_model():

    """
    Function that loads pretrained VGG model to extract features from images

    Returns
    -------
    model: Keras Model
        Pretrained VGG19 Model
    """
    #load pretrained VGG model
    model = VGG16(
        include_top=True,
        weights='imagenet'
    )
    model = Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    return model


def predict_data(image_dict):

    """
    Function to predict feature representation for all elements of the dataset

    Parameters
    ----------
    image_dict: dict
        Dictionary with image filenames as keys and array representation of the image as values
    
    Returns
    -------
    output: dict
        Dictionary with image filenames as keys and feature representation of the images as values
    """
    model = load_prebuilt_model()
    #make prediction for image arrays in dictionary
    output = {k: model.predict(v) for k,v in image_dict.items()}
    return output


if __name__ == '__main__':

    #loading images
    print("Loading and preprocessing images from directory...")
    image_dict = load_preprocess_images()
    #extracting features
    print("Extracting features from images...")
    output = predict_data(image_dict)
    #dumping output to pickle file
    print("Pickling image features...")
    pickle.dump(
        output,
        open("image_features.p", "wb")
    )
    print("Finished...")
    
