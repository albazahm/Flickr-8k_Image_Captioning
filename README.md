# **Image Captioning Using Deep Learning**

This project explores the use of a deep learning for image captioning. This will be accomplished by using merged architecture that combining a Convolutional Neural Network (CNN) with a Long-Short-Term-Memory (LSTM) network.

The goal is to be able to create an automated way to generate captions for a given image. A caption of an image gives insight as to what is going on or who is present in an image. It is a useful starting point for automated Q/A with images/video. For example, further development could result in a bot which can answer questions about an image.

---

## **About the Dataset**

The dataset used for this project is the Flickr-8k dataset. It consists of 8000 different images along with their captions, sourced from Flickr. Each image in turn has up to 5 different captions that describe the details of the image.

Citation:

    Hodosh, Micah, Peter Young, and Julia Hockenmaier. "Framing image description as a ranking task: Data, models and evaluation metrics." Journal of Artificial Intelligence Research 47 (2013): 853-899.

Examples from the dataset:

!['Examples From Dataset'](./README_figures/README_fig1.png)

---

## **Project Outline**

1. **Image preprocessing and feature extraction** ([code](./preprocess_images.py))

    This was accomplished using the pre-trained VGG16 model available in the Keras library, along with its utility function for preprocessing. The output of the first fully connected layer in the VGG model was used as the feature for a given images.

2. **Text preprocessing and feature extraction** ([code](./preprocess_text.py))

    This step involved writing functions that lowercased, removed punctuation and numeric characters, tokenized the captions, adding indicators for the beginning and end of a seqeunces, and creating sequences from those indices for that caption. For example: given the following caption:

        "A little girl playing with a ball."

    The expected output after preprocessing is:

        "*start*, a, little, girl, playing, with, a, ball, *end*"

    The expected output creating sequences:

        INPUT                                                        OUTPUT

        *start*                                                      a
        *start*, a                                                   little
        *start*, a, little                                           girl
        *start*, a, little, girl                                     playing
        *start*, a, little, girl, playing                            with
        *start*, a, little, girl, playing, with                      a
        *start*, a, little, girl, playing, with, a                   ball
        *start*, a, little, girl, playing, with, a, ball             *end*

    In this manner, a 6 word-sentence gets transformed 8 inputs and 8 outputs. Note that each image will come with 5 different captions, each of which must be transformed in this way.

    Additionally, the model does not understand the words in this current form. They must be encoded as numbers. The unique words in the entire caption corpus were extracted, sorted and each assigned an index.

    Next, the inputs must be standardized to a common length for the model. The input sequences of indices were padded with zeros to a length equal to that of the longest caption in the corpus.

    Finally, the output sequences were one-hot-encoded to allow for a multi-class prediction by the model.

3. **Model Building** ([code](./deep_learning_model.py))

    The model using a merged architecture. As such, there are two input streams, one corresponding to the image features and one to the text features.

    The image stream involves a BatchNormalization layer, then a Dropout layer followed by a ReLu activated Dense layer.

    The text stream involves an Embedding layer, followed by a Dropout Layer and a LSTM layer.

    The outputs from each of the streams above is of the same shape. They get added before being passed into another ReLu activated Dense Layer and finally the SoftMax activated output layer to produced the label.

    [View Model Summary](./README_figures/README_fig2.png)

4. **Generator** ([code](./deep_learning_model.py))

    Since the data was too big too fit into memory, a generator was used to load the data in batches. Keras is compatible with generators when training the model so that facilitated training within memory at the expense of slower training time.

5. **Prediction and Evalution** ([code](./evaluation.py), [notebook](./Image_Captioning.ipynb))

    For the prediction stage, the model is given an image feature and an array with index correspoding to the start indicator. It is then tasked with predicting the next word. The index of the predicted word is appended to the previous input and a new prediction is generated. This is repeated until the end indicator is predicted. At that point, the prediction is complete.

    The predictions were evaluated on items from within the dataset as well as on unseen images to determine its performance.
