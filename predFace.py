from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

jpg_name = ""
model_file = './hanaza.h5'

model = load_model(model_file)

img_path = "./testNadeko.jpg"
img = img_to_array(load_img(img_path, target_size=(28,28)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label = ['kuroneko','kuroneko']
pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print('name:', pred_label)
print('score:', score)