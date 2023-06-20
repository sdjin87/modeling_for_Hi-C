#Tool for Cis-platin resistance prediction using Hi-C matrix data(v1.0)
import sys
data_dir=sys.argv[1]

#ex="plot/test/sensitive/a2780_chm13_deg_chr19_760000-770000.txt.png"
#importing packages
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#generate dataset from input data

#model importing
model=tf.keras.models.load_model("hic_resistance_model2.h5") #acc : 87~91%

#prediction
img = Image.open(data_dir).convert('RGB').resize((200, 200), Image.ANTIALIAS)
img = np.array(img)
img = img.astype('float32') / 255.

plt.imshow(img, interpolation='nearest');

test_num = img.reshape((1,200, 200, 3))
y_prob = model.predict(test_num, verbose=1) 
predicted = y_prob.argmax(axis=-1)
class_map = {
    0:'cis-platin resistant',
    1:'cis-platin sensitive', 
}
print('The HI-C data is ', class_map[predicted[0]]) 
