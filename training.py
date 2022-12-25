from itertools import chain
#libraries needed for NLP
import nltk
nltk.download('punkt')


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
words = sorted(set(list(chain.from_iterable([pattern for pattern, response in patterns_and_responses]))))

# Stem the words and lowercase them
words = [stemmer.stem(token.lower()) for token in words]
# Extract the unique response types
types = sorted(set(response for _, response in patterns_and_responses))

#print(types)
#print(words)
# Create the training data

training = []

# Create an empty array for output
output_empty = [0] * len(types)

# Create training set bag of words for each sentence
for doc in patterns_and_responses:
    # Initialize bag of words
    bag = [1 if stemmer.stem(w.lower()) in bag else 0 for w in words]
    
    # Output is 1 for current tag and 0 for the rest of other tags
    output_row = [1 if c == doc[1] else 0 for c in types]
    
    training.append([bag, output_row])

# Shuffle the training data and convert it to a numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the training data into input and output lists
train_x, train_y = list(training[:, 0]), list(training[:, 1])

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

