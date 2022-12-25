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

ERROR_THRESHOLD = 0.0
def classify(sentence):
  #generate probabilities from the model
  results = model.predict([bow(sentence, words)])[0]
  
  #filter out prediction below a threshold
  results = [[i,r] for i, r in enumerate(results) if r> ERROR_THRESHOLD]
  
  #sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list=[]
  
  for r in results:
    return_list.append((classes[r[0]], r[1]))
  
  #return tuple of intent and probability
  return return_list


def response(phrase):
  results = classify(phrase)
  
  if results:
    # Find the first matching intent
    while results:
      for i in intents['intents']:
        if i['tag'] == results[0][0]:
          # Return a random response from the matching intent
          return print(random.choice(i['responses']))
      results.pop(0)
