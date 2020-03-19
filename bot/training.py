"""
Script that trains a neural network on NL's dialogue

To-do:
- Hook up to youtube data
"""


import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

if __name__ == "__main__":

    # Load in transcripts
    with open('test_transcripts/afterbirthplus_transcripts.txt') as file:
        raw_text = file.read()

    # Processing. (youtube transcripts dont seem to need much)
    text = raw_text.replace('\n', ' ')

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
    seq_length = 140
    for i, char in enumerate(text):
        #print(i*100/len(text))
        if i >= seq_length:
            sub_string = text[i-seq_length:i]
            x_instance = [char_to_int[character] for character in sub_string]
            y_instance = char_to_int[char]
            X_raw.append(np.array(x_instance))
            y_raw.append(np.array(y_instance))

    X = np.reshape(X_raw, (len(X_raw), seq_length, 1))
    X = X/len(chars)  # Normalising
    y = np_utils.to_categorical(y_raw)

    # Build model and store architecture as json
    model = Sequential()
    model.add(layers.LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(layers.Dropout(0.4))
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
