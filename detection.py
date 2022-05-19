from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
import uvicorn   
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
  
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
  
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('index.html')
    


MODEL = tf.keras.models.load_model("models/1")




CLASS_NAMES = ['Amaltas',
 'False ashoka',
 'Mauritious hemph',
 'bougainvillea glabra',
 'daisy',
 'dandelion',
 'rose',
 'sunflower',
 'thuja']
    

        
        

def predict_image(img_path):
    test_image = image.load_img(img_path, target_size=(256,256) )
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image,axis=0)

    predictions = MODEL.predict(test_image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return predicted_class, confidence

    



   
 
   
@app.route('/', methods=['GET', 'POST'])
def getvalue():
    if request.method == 'POST':
       img = request.files['imgfile']
       img_path = "static/images/" + img.filename
       img.save(img_path)
       output, scname = predict_image(img_path)
       print(output)

    return render_template('result.html', o=output, scname=scname, fimage=img.filename)  


# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)