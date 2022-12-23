#libraries needed for NLP
import nltk
nltk.download('punkt')
import json
from itertools import chain

# Choose a stemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

#Libraries needed for TensorFlow processing
import tensorflow as tf
import numpy as np
import random

# Load the intents data from the JSON file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Flatten the list of patterns and responses into a single list
patterns_and_responses = [(nltk.word_tokenize(pattern), intent['tag']) for intent in intents['intents'] for pattern in intent['patterns']]

# Extract the unique words from the patterns
words = sorted(set(chain.from_iterable(patterns_and_responses)))

# Stem the words and lowercase them
words = [stemmer.stem(token.lower()) for token in words]

# Extract the unique response types
types = sorted(set(response for _, response in patterns_and_responses))

# Create the training data
training = []
output = []
output_empty = [0] * len(types)
for pattern, response in patterns_and_responses:
    bag = [1 if stemmer.stem(token.lower()) in words else 0 for token in pattern]
    output_row = list(output_empty)
    output_row[types.index(response)] = 1
    training.append((bag, output_row))

# Shuffle the training data and convert it to a numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the training data into input and output lists
train_x, train_y = training[:, 0], training[:, 1]

# Reset the TensorFlow graph
tf.keras.backend.clear_session()

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(len(train_x[0]),)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=1000, batch_size=8)

# Save the model
model.save('model.h5')

