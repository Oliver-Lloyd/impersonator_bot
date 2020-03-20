from keras.utils import np_utils
import numpy as np
import json

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
        print(i * 100 / len(text))
        if i >= seq_length:
            sub_string = text[i - seq_length:i]
            x_instance = [char_to_int[character] for character in sub_string]
            y_instance = char_to_int[char]
            X_raw.append(x_instance)
            y_raw.append(y_instance)

    X = np.reshape(X_raw, (len(X_raw), seq_length, 1))
    X = X / len(chars)  # Normalising
    np.save('NL_X.npy', X)
    y = np_utils.to_categorical(y_raw)
    np.save('NL_y.npy', y)
