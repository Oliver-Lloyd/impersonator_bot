"""
Script that trains a neural network on NL's dialogue and then generates sentences.

To-do:
- Transfer learning instead of building own model from scratch?
- Hook up to youtube data
"""


import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
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

# Create X and y
X_raw = []
y_raw = []
seq_length = 100
for i, char in enumerate(text):
    if i >= seq_length:
        sub_string = text[i-100:i]
        x_instance = [char_to_int[character] for character in sub_string]
        y_instance = char_to_int[char]
        X_raw.append(x_instance)
        y_raw.append(y_instance)

# reshape X to be [samples, time steps, features]
X = np.reshape(X_raw, (len(X_raw), seq_length, 1))
# normalize
X = X / float(len(chars))

y = np_utils.to_categorical(y_raw)


# Build model
model = Sequential()
model.add(layers.LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Do not need test data as we are trying to learn from the entire dataset and accurate predictions are not important.
# Minimising the loss function is all that matters.

# Use checkpoints to save weights each time improvement in loss is achieved
file_path = "alice_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, EarlyStopping(patience=3, monitor='loss')]

model.fit(X, y,
          epochs=100, batch_size=256,
          callbacks=callbacks_list)
