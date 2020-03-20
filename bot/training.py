from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import load


if __name__ == "__main__":
    X = load('NL_X.npy')
    y = load('NL_y.npy')

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
