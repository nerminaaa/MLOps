import os 

import numpy as np
import matplotlib.pyplot as plt
from   tensorflow.keras.utils import load_img

def test_model(model, folder_path, class_names, target_size=(124, 124)):
    for image in os.listdir(folder_path):
        test_image = load_img(f'{folder_path}/{image}', target_size=target_size)
        
        # Show the image
        plt.imshow(test_image, interpolation='spline16')
        plt.xticks([]), plt.yticks([])  # Hide ticks
        plt.show()

        # Prepare the image for prediction
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        for i, label in enumerate(class_names):
            print(f"{label} ==> {result[0][i]*100:.2f}%")
