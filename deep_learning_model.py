import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import random
import pickle

def create_model(image_embedding_size, vocab_size, maxlen):

    """
    Function to create a Keras neural network. Given a sequence of word indicies, the model will predict
    the index of the next word in the sequence. It uses text information from earlier parts of the sequence
    alongside image features that were extracted from the captioned pictures.

    Parameters
    ----------
    image_embedding_size: int
        The size of the feature representation of the images
    vocab_size: int
        The number of unique words among all captions
    maxlen: int
        The maximum length of a sequence

    Returns
    -------
    model: Keras model object
        A Keras model that takes feature representation of an image as well as a text sequence as inputs and makes
        a prediction as to what is the next word in that sequence
    """
    #image branch
    image_input_layer = Input(shape=(image_embedding_size, ))
    image_drop_layer = Dropout(0.5)(image_input_layer)
    image_dense_layer = Dense(128, activation='relu')(image_drop_layer)
    
    #text branch
    text_input_layer = Input(shape=(maxlen, ))
    text_em_layer = Embedding(vocab_size, 256, mask_zero=True)(text_input_layer)
    text_dropout_layer = Dropout(0.5)(text_em_layer)
    text_lstm_layer = LSTM(128)(text_dropout_layer)
    
    #merging image and text branches
    add_layer = Add()([image_dense_layer, text_lstm_layer])
    fc_layer = Dense(128, activation='relu')(add_layer)
    output_layer = Dense(vocab_size, activation='softmax')(fc_layer)

    #compiling model
    model = Model(
        inputs=[image_input_layer, text_input_layer], 
        outputs=output_layer)

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam')

    return model


def generate(images, text, target, batch_size, vocab_size, seed=42):

    """
    Function that yields batches of images, text and labels for training a model when
    data is too big to fit in memory

    Parameters
    ----------
    images: list
        List of VGG-16 feature predictions for images
    text: Numpy array
        Array of text sequences corresponding to captions
    target: list
        List of labels corresponding to the next word in the caption
    batch_size: int
        The number of samples per batch to generate
    vocab_size: int
        The number of unique words in the target vocabulary

    """
    while 1:
        #zipping inputs and outputs together
        data = list(zip(images, text, target))
        #shuffling inputs and outputs together every epoch
        random.seed(seed)
        random.shuffle(data)
        #initilizing count for number of observations in a batch
        batch_count = 0
        #initializing lists for image, text, and target batches
        image_batch = []
        text_batch = []
        target_batch = []
        #looping over the zipped lists and sampling observations for each batch
        for (image_sample, text_sample, target_sample) in data:
            #one hot encoding target
            target_sample = to_categorical(target_sample, num_classes=vocab_size)
            #sampling observations for each batch
            image_batch.append(image_sample)
            text_batch.append(text_sample)
            target_batch.append(target_sample)
            #increasing batch count
            batch_count += 1
            #if batch count reaches the batch_size, the generator will yield the batch
            if batch_count==batch_size:
                image_batch = np.array(image_batch).reshape(batch_size, image_sample.shape[1])
                text_batch = np.array(text_batch)
                target_batch = np.array(target_batch).reshape(batch_size, vocab_size)
                yield [image_batch, text_batch], target_batch
                image_batch = []
                text_batch = []
                target_batch = []
                batch_count = 0
        

if __name__ == '__main__':

    BATCH_SIZE = 64
    EPOCHS = 10

    #load image features
    image_features = pickle.load(open('image_features.p', 'rb'))
    #load sequences and filenames
    X, y, filenames = pickle.load(open('text_features.p', 'rb'))
    #load mappings
    word_to_ind, ind_to_word = pickle.load(open('mappings.p', 'rb'))

    #match image features to filenames
    img_array = list(map(image_features.get, filenames))
    #extract size of image features
    image_embedding_size = img_array[0].shape[1]
    #extract maxlen
    maxlen = len(X[0])
    #extract vocab_size
    vocab_size = len(np.unique(y)) + 1
    #calculate step size for generator
    steps = int(len(X)/BATCH_SIZE)
    #initialize generator
    generator = generate(
        images=img_array,
        text=X,
        target=y,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size
    )
    #create deep learning model
    model = create_model(
        image_embedding_size=image_embedding_size,
        vocab_size=vocab_size,
        maxlen=maxlen
    )
    #fit model on generator 
    history = model.fit(
        generator,
        steps_per_epoch=steps,
        epochs=EPOCHS
    )

    #set parameters to untrainable
    for layer in model.layers:
        layer.trainable = False
    #save the model
    model.save('image_captioner')