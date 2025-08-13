import tensorflow as tf

def load_model():
    """Load pre-trained emotion recognition model"""
    model = tf.keras.models.load_model('models/emotion_model.h5')
    return model
