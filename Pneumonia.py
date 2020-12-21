import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

pneumonia_model = load_model('pneumonia/pneumonia.h5')
UPLOAD_FOLDER = 'uploads'

def preprocessing(path):
    data = image.load_img(path, target_size=(224, 224, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    #with graph.as_default():
    predicted =  pneumonia_model.predict_classes(data)[0][0]
    return predicted


def predict(file):
    file_name = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_name)
    result = preprocessing(file_name)
    print(result)
    predicted_class = np.asscalar(np.argmax(result, axis=1))

    return predicted_class

