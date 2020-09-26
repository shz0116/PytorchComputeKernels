
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3))

