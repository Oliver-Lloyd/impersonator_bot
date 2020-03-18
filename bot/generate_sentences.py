from tensorflow import keras
from os import listdir

# Load model architecture
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = keras.models.model_from_json(loaded_model_json)

# Load best weights
weight_paths = [file for file in listdir() if file.endswith('hdf5')]
lowest_loss = 1e9
path_to_lowest_loss = None
for path in weight_paths:
    loss = float(path.split('.hdf5')[0][-6:])
    if loss < lowest_loss:
        lowest_loss = loss
        path_to_lowest_loss = path
model.load_weights(path_to_lowest_loss)
