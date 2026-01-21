import numpy as np
from tensorflow.keras.models import load_model
from  tensorflow.keras.preprocessing import image

import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    def predict(self):
        model=load_model("artifacts/prepare_base_model/base_model_updated.h5")

        imagename=self.filename
        test_image.load_img(imagename,target_size=(224,224))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image, axis=0)
        result=np.argmax(model.predict(test_image), axis=1)
        

        if result==1:
            prediction='healthy'
            return [{"image":prediction}]

        else:
            prediction='diseased'
            return [{"image":prediction}]
        
        
