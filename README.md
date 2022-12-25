# Contextual-Chatbot

This code is implementing a natural language processing (NLP) chatbot using TensorFlow. The chatbot is trained to understand and respond to user input based on a set of predefined patterns and responses stored in a JSON file called 'intents.json'.

<p align="center">
  <img src="https://github.com/Jose-M-Ramos/Contextual-Chatbot/blob/main/chat-bot.jpg" width="600" height="400">
</p>


I did this project to put into practice the fundamentals of Natural Language Processing and Neural Networks. I discuss the limitations and the great scope for improvement at the end of this file.



## Dependencies

This chatbot requires the following libraries:

* nltk: Is a library that provides tools for working with human language data (text). It is used in this chatbot to tokenize user input into individual words.

* tensorflow: It is used in this chatbot to build and train the neural network that is used to predict the intent of user input.

* numpy: It is used in this chatbot to manipulate the training data and the input from the user.

* keras: Is a high-level neural networks API written in Python that runs on top of other lower-level libraries such as TensorFlow. It allows users to define, compile, and fit complex machine learning models with minimal code. It is used to build a machine learning model with a feedforward neural network architecture

* json: It allows for the parsing and manipulation of JSON (JavaScript Object Notation) data. It is used in this chatbot to load the intents.json file, which contains the patterns and responses used to train the chatbot.

* pickle: It is used in this chatbot to save the training data to a file, and to later load it for use in the chatbot.



## Data

The intents.json file is a JSON file that contains data for training a chatbot. The file contains a list of "intents", each of which represents a category of user input that the chatbot should be able to recognize and respond to. Each intent has the following fields:

tag: A label indicating the category of the intent.
patterns: A list of natural language text patterns that belong to this intent.
responses: A list of responses that the chatbot can use to respond to user input belonging to this intent.
context_set (optional): A string indicating a context that should be set after responding to user input belonging to this intent.
context_filter (optional): A string indicating a context that should be used to filter the list of responses for user input belonging to this intent.
The intents.json file is used to train a machine learning model to classify user input into the predefined categories (or "intents"). The chatbot can then use the trained model to classify user input and respond with a suitable response from the predefined set of responses.



## Training

The input data is first preprocessed by extracting the unique words from the text patterns and stemming (truncating) them to their base form. These words are then used to create a bag of words representation of the text data, which is a numerical representation of a text document in which each word is represented by a binary value indicating whether it appears in the document or not. The bag of words representation is then used as input to the feedfordward neural network, which is trained to classify the input data into the predefined categories and respond with a suitable response from a predefined set of responses.


## How to use?

```python
# Import the necessary libraries
import nltk
import tensorflow as tf
import numpy as np
import json
import random
import pickle

# Load the intents data from the JSON file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Load the data from the pickle file
data = pickle.load(open("training_data", "rb"))
words = data['word']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Use the classify and response functions to generate a response
phrase = "Do you take credit cards?"
print(classify(phrase))
print(response(phrase))
```

## Limitations

There are several limitations to the code you provided that may prevent it from working effectively as a chatbot. Here are a few examples:

1. The training data may be insufficient to accurately classify a wide range of user input. The chatbot's performance may degrade if it receives input that is significantly different from the patterns it was trained on.

2. The chatbot may not be able to handle context-dependent conversations. For example, if a user asks a question that depends on a previous question or statement made by the chatbot, the chatbot may not be able to understand the context and provide a suitable response.

3. The chatbot may not be able to handle variations in the wording or structure of user input and misspellings or typos. For example, if a user asks a question in a different way than the patterns it was trained on, the chatbot may not be able to recognize the intent of the user's input and provide a suitable response.

4. The chatbot may not be able to handle multiple user inputs at once or handle input that contains multiple intents.

Overall, the code you provided may not be sufficient to build a fully functional chatbot that can handle a wide range of user input and respond in an appropriate and natural way. Additional work and improvements may be necessary to address these limitations.

## Areas of Improvements

There are several improvements that could be made to the code you provided to make it more effective as a chatbot:

1. Increase the size and diversity of the training data to better cover a wide range of user input and improve the chatbot's ability to classify user input accurately.

2. Add support for handling context-dependent conversations by using a context-aware neural network architecture or by incorporating additional information about the conversation history into the input data.

3. Use a neural network architecture that is better suited for handling variations in the wording or structure of user input, such as a recurrent neural network (RNN) or a transformer.

4. Add spelling correction or typo handling capabilities to the chatbot.

5. Add support for multiple languages by using a multilingual neural network model or by training separate models for each language.

Ultimately, the choice of neural network architecture will depend on the specific requirements and goals of the chatbot and the available training data. 
