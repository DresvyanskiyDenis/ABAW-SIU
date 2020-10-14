import tensorflow as tf



def CNN_1D_model(input_shape, num_classes):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=128, kernel_size=20, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=8))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, activation='relu', padding='same',))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(256, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def LSTM_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(input_shape=input_shape, mask_value=0.))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def sequence_to_sequence_regression_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=32, kernel_size=20, strides=1, activation='relu',
                                     padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation='relu', padding='same', ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    print(model.summary())
    return model

if __name__ == "__main__":

    input_shape=(100, 23)
    model=LSTM_model(input_shape, 8)
