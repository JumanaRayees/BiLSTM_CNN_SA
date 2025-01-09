# BiLSTM_CNN_SA

# Project Goal:

The code is designed for sentiment analysis of movie reviews, classifying them as either positive or negative. It involves building and training a hybrid neural network model that combines Bidirectional LSTM and Convolutional Neural Network (CNN) layers for text processing.

# Code Breakdown:

Environment Setup and Data Loading:

The code starts by installing the nltk, tensorflow-text and tensorflow libraries, which are necessary for natural language processing and building neural networks.
It downloads the stopwords corpus from nltk, which is a list of common words that are often removed during text processing.
It then unzips a file called txt_sentoken.zip into the /tmp directory. This archive likely contains the movie review dataset.
Utility functions load_doc and clean_doc are defined for loading text from files and cleaning it by removing punctuation, non-alphabetic characters, and stop words.
Vocabulary Creation:

The add_doc_to_vocab function processes each document, cleans the text, and adds the tokens to a vocabulary.
The process_docs function walks through the positive and negative review directories, adding each document to the vocabulary using the previous function.
The code then keeps words that appear at least twice, to help remove words that occur too rarely.
The vocabulary (list of relevant words) is saved to a file named vocab.txt.
Data Preparation:

The vocab.txt file is loaded back into memory, which contains all the valid words.
The clean_doc function is redefined to remove words that are not in the vocabulary.
The process_docs function is redefined to load, clean, and return a list of cleaned documents.
Positive and negative reviews are loaded, cleaned and combined into one list of train documents.
A tokenizer is fit on the training documents to be used to convert words to integers.
The training documents are then converted to sequences of integers based on the tokenizer, and these sequences are padded so that they all have the same length.
The labels for training data are defined. 0 for negative and 1 for positive reviews.
Model Definition:

1- The code defines a neural network model that uses an Embedding layer, a Bidirectional LSTM layer, a Reshape layer, a 2D Convolutional layer, a MaxPooling2D layer, a Dropout layer, a Flatten layer, and finally a Dense layer to make its predictions.
Embedding Layer: Converts words to vector representations.
Bidirectional LSTM: Processes the sequence in both forward and backward directions, capturing context.
Reshape Layer: Adds a dimension for the CNN.
CNN: Extracts local features from the sequence.
MaxPooling: Reduces dimensionality and highlights important features.
Dropout: Prevents overfitting.
Flatten: Converts 2D data to 1D.

Dense: Makes the final classification based on extracted features.
2- The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as a metric.
Model Training:

The model is trained on the prepared training data using the fit method.
An EarlyStopping callback is used to prevent overfitting.
Model Evaluation:

3- The training set is split again into training and validation sets, for performance evaluation.
The model makes predictions on the validation set.

4- Several metrics (accuracy, precision, recall, F1-score) are calculated.
A confusion matrix is generated and visualized.
Training and validation loss curves are plotted.



