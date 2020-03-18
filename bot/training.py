"""
Script that trains a neural network on NL's dialogue

To-do:
- Hook up to youtube data
"""


import numpy as np
import re
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

if __name__ == "__main__":

    # Get toy data
    with open('test_transcripts/alice_wonderland.htm') as file:
        raw_text = file.read()

    # Select correct text (initial processing here is calibrated for Alice data, will need reworking for NL)
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
    text = text.replace('_', '')
    spaces = re.compile(' {2,}')
    text = re.sub(spaces, ' ', text)

    # Create and store dicts for converting
    chars = set(text)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    json_raw = json.dumps(char_to_int)
    f = open("char_to_int.json", "w")
    f.write(json_raw)
    f.close()

    int_to_char = dict((i, c) for i, c in enumerate(chars))
    json_raw = json.dumps(int_to_char)
    f = open("int_to_char.json", "w")
    f.write(json_raw)
    f.close()

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
    X = np.reshape(X_raw, (len(X_raw), seq_length, 1))
    X = X/len(chars)  # Normalising
    y = np_utils.to_categorical(y_raw)

    # Build model and store architecture as json
    model = Sequential()
    model.add(layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(256))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Using checkpoints to save weights each time improvement in loss is achieved
    file_path = "alice_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, EarlyStopping(patience=3, monitor='loss')]

    # Run the training
    model.fit(X, y,
              epochs=100, batch_size=256,
              callbacks=callbacks_list)
