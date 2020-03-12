"""
Script that trains a neural network on NL's dialogue and then generates sentences.

To-do:
- Transfer learning instead of building own model from scratch
- Hook up to youtube data
"""


import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Get toy data
with open('test_transcripts/alice_wonderland.htm') as file:
    raw_text = file.read()

# Select correct text
start_string = 'Down the Rabbit-Hole</h3>\n\n<p>'
start_index = raw_text.find(start_string) + len(start_string)
end_string = '</p>\n\n<p>End of the Project Gutenberg Etext'
end_index = raw_text.find(end_string)
trimmed_text = raw_text[start_index:end_index]

# Remove special characters
text = trimmed_text.replace('\n\n', '\n').replace('\n', ' ')
text_modifiers = re.compile('<.*?>')
text = re.sub(text_modifiers, '', text)
text = text.replace(',)', ')')
text = text.replace('*&nbsp;', '')
text = text.replace('&nbsp;', '')
text = text.replace('*', '')
spaces = re.compile(' {2,}')
text = re.sub(spaces, ' ', text)

chars = set(text)
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(split_lines)
sequences = tokenizer.texts_to_sequences(split_lines)
word_index = tokenizer.index_word
num_words = len(word_index)

# Create X and y
X = []
y_raw = []
x_len = 4
for seq in sequences:
    for ind in range(x_len, len(seq)):
        sub_sequence = seq[ind - x_len:ind + 1]
        X.append(sub_sequence[:-1])
        y_raw.append(sub_sequence[-1])
X = np.array(X)
# Encode y one-hot
y = np.zeros((len(y_raw), num_words + 1))
for row, column in enumerate(y_raw):
    y[row, column] = 1

# Make train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential()
model.add(layers.Embedding(input_dim=num_words+1, input_length=4,
                           output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_words+1, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(np.array(X_train), y_train,
          batch_size=256, epochs=512,
          callbacks=[EarlyStopping(patience=5)])

model.evaluate(X_test, y_test)

# Generate predictions from the model
words_to_add = 10
og_phrase = 'ryan and dan are'
for i in range(words_to_add):
    phrase = og_phrase.split(' ')[i:i+4]
    encoded = []
    for word in phrase:
        if word not in word_index.values():
            raise ValueError('The word "%s" is not in the transcript.' % word)
        for ind, string in word_index.items():
            if word == string:
                encoded.append(ind)
                continue

    prediction_probs = model.predict(np.array([encoded]))
    predicted_word = word_index[np.argmax(prediction_probs[0])]
    og_phrase = og_phrase + ' ' + predicted_word
    print(og_phrase)