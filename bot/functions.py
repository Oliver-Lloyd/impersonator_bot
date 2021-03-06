def load_model(model_path='./model.json', weights_directory='.'):
    from tensorflow import keras
    from os import listdir

    # Load architecture
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json)

    # Load best weights
    weight_paths = [file for file in listdir(weights_directory) if file.endswith('hdf5')]
    lowest_loss = 1e9
    path_to_lowest_loss = None
    for path in weight_paths:
        loss = float(path.split('.hdf5')[0][-6:])
        if loss < lowest_loss:
            lowest_loss = loss
            path_to_lowest_loss = path
    model.load_weights(path_to_lowest_loss)

    return model


def process_x(requested_text, char_to_int_path='char_to_int.json'):
    import json
    import numpy as np

    # Load in converter dict
    with open(char_to_int_path, "r") as f:
        char_to_int = json.load(f)

    # Convert to list (in future may want to handle multiple requests at once)
    if type(requested_text) != list:
        requested_list = [requested_text]
    else:
        requested_list = requested_text

    x = []
    for request in requested_list:
        processed = []
        for char in request:
            value = char_to_int[char]
            processed.append(value)
        x.append(np.array(processed))

    x = np.array(x)
    x = np.reshape(x, (len(x), len(requested_text), 1))
    return x


def generate(input_text, max_words=10, max_chars=140, model_path='./model.json', char_to_int_path='./char_to_int.json', int_to_char_path='./int_to_char.json'):
    # need to add functionality to end at the first space that occurs after num_chars
    import json
    from numpy import argmax
    with open(char_to_int_path, "r") as f:
        char_to_int = json.loads(f.read())
    with open(int_to_char_path, "r") as f:
        int_to_char = json.loads(f.read())
    og_text = input_text
    og_spaces = og_text.count(' ')
    model = load_model(model_path, '.')
    input_len = model.input_shape[1]
    for _ in range(max_chars):
        current_spaces = og_text.count(' ')
        if current_spaces-og_spaces >= max_words:
            return og_text
        if len(og_text) < input_len:
            text = og_text
            while len(text) != input_len:
                text = ' ' + text
        elif len(og_text) > input_len:
            text = og_text[-input_len:]
        else:
            text = og_text

        x = process_x(text, 'char_to_int.json')
        predict_probs = model.predict(x)
        pred_index = argmax(predict_probs)
        pred_character = int_to_char[pred_index]
        og_text = og_text + pred_character

    return og_text
