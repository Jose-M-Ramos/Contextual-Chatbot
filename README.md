# Contextual-Chatbot

This code is implementing a natural language processing (NLP) chatbot using TensorFlow. The chatbot is trained to understand and respond to user input based on a set of predefined patterns and responses stored in a JSON file called 'intents.json'.

<p align="center">
  <img src="https://github.com/Jose-M-Ramos/Contextual-Chatbot/blob/main/chat-bot.jpg" width="600" height="400">
</p>


## Dependencies

This chatbot requires the following libraries:

* nltk: is a library that provides tools for working with human language data (text). It is used in this chatbot to tokenize user input into individual words.

* tensorflow: It is used in this chatbot to build and train the neural network that is used to predict the intent of user input.

* numpy: It is used in this chatbot to manipulate the training data and the input from the user.

* tflearn: is a library built on top of TensorFlow that provides additional functionality for building and training machine learning models. It is used in this chatbot to build the neural network and to train it on the training data.

* json: allows for the parsing and manipulation of JSON (JavaScript Object Notation) data. It is used in this chatbot to load the intents.json file, which contains the patterns and responses used to train the chatbot.

* pickle: It is used in this chatbot to save the training data to a file, and to later load it for use in the chatbot.



## Data

## Training

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

