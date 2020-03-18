def load_model(model_path, weights_directory):
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


def process_x(requested_text, path_to_dict_json):
    import json
    import numpy as np

    # Load in converter dict
    with open(path_to_dict_json, "r") as f:
        char_to_int = json.load(f)

    # Convert to list
    if type(requested_text) != list:
        requested_list = [requested_text]
    else:
        requested_list = requested_text

    X = []
    for request in requested_list:
        processed = []
        for char in request:
            value = char_to_int[char]
            processed.append(value)
        X.append(processed)

    X = np.array(X)
    return X