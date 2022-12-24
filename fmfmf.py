import nltk
import numpy as np
import random
import json
import pickle
import training

# Load the intents data from the JSON file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Save the words, classes, train_x, and train_y variables to a pickle file
pickle.dump({'word': Words, 'classes': Types, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

# Load the data from the pickle file
data = pickle.load(open("training_data", "rb"))
words = data['word']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

def clean_up_sentence(phrase):
  # Tokenize the phrase
  sentence_words = nltk.word_tokenize(phrase)
  # Stem and lowercase each word
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

def bow(phrase, words, show_details=False):
  # Tokenize the phrase
  sentence_words = clean_up_sentence(phrase)
  # Generate the bag of words
  bag = [0] * len(words)
  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("Found in bag: %s" % w)
  return np.array(bag)

threshold = 0.0
def classify(phrase):
  # Generate probabilities from the model
  results = model.predict([bow(phrase, words)])[0]
  
  # Filter out predictions below the threshold
  results = [[i, k] for i, k in enumerate(results) if k > threshold]
  
  # Sort the results by probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  
  # Create a list of (tag, probability) tuples
  for k in results:
    return_list.append((classes[k[0]], k[1]))
  
  # Return the list of tuples
  return return_list

def response(phrase):
  results = classify(phrase)
  
  if results:
    # Find the first matching intent
    for i in intents['intents']:
      if i['tag'] == results[0][0]:
        # Return a random response from the matching intent
        return random.choice(i['responses'])

if __name__ == "__main__":
    response()
