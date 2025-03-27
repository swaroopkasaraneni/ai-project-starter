import numpy as np
import tensorflow as tf

class SimpleAIModel:
    def __init__(self, input_dim=10, output_dim=1):
        self.model = self._build_model(input_dim, output_dim)
    
    def _build_model(self, input_dim, output_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X, y, epochs=100, batch_size=32):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)