import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

width, height = 150, 150
model = '.\model\model.h5'
cnn = load_model(model)
model_weights = '.\model\weights.h5'
cnn = load_model(model)
cnn.load_weights(model_weights)

print(model)

def predict(file):
    x = load_img(file, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Prediction: Catedral")
    elif answer == 1:
        print("Prediction: Palacio de Gobierno")
    elif answer == 2: 
        print("Prediccion: Estatua fudadores Tec2")
        el fi

    return answer

predict("scene1800161.jpg")