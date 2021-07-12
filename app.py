import numpy as np
from flask import Flask,render_template,request
import tensorflow
from tensorflow import keras
from keras import models
from keras.models import load_model
from keras.preprocessing import image
import os
import skimage.io
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.io import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import random
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json



# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_hand_gt_big.h5") 



# from werkzeug import secure_filename
app=Flask(__name__)

UPLOAD_PATH='./static/image_stored'

#model=load_model('model_hand_gt_big.h5')
# dirapp = r'D:/Videh_Acads/Machine_Learning/GBD_Project/temp/'
# os.chdir(dirapp)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_hand_gt_big.h5")    

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            img_location=os.path.join(UPLOAD_PATH,image_file.filename)
            #img_location = UPLOAD_PATH+ '\\'+ image_file.filename
            image_file.save(img_location)
            IMG_HEIGHT = 1084               #1084, 572
            IMG_WIDTH = 1084                #1084, 572
            IMG_CHANNELS = 3
            #img = mpimg.imread('1.tif')
            img = imread(img_location)
            #img = image_file
            img_orig = resize(img, (IMG_HEIGHT, IMG_WIDTH))

            #if the image is coloured
            if img.ndim == 3:
                img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img[:,:, np.newaxis]
                orig = img.copy()
            #or if its gray-scaled    
            else:
                img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)).astype(np.uint8) 
                img = img[:,:, np.newaxis] 
                orig = img.copy()
    

            img = img[np.newaxis,:,:,:]
            #print(img.shape)

            # load json and create model
            #root_dir = r'C:/Users/Videh Aggarwal/Python/' 
            #os.chdir(root_dir)
            
           
            # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
                  #model_hand_gt_big.h5           #model_countour_gabor_900.h5
            #print("Loaded model from disk")
                
            # evaluate loaded model on test data
            
            pred = loaded_model.predict(img, verbose=1)
            mean = pred.mean()
            result_1 = (pred> mean).astype(np.uint8)
            result_1 = result_1*255
            '''
            optional dilation:::
            
            '''
            TRAIN_PATH= r'D:/Videh_Acads/Machine_Learning/data/Sir_Micrographs/micrographs Inconel-20200827T122844Z-001/Train_Images/'
            directory = r'D:/Videh_Acads/Machine_Learning/GBD_Project/temp/static/predictions/'
            os.chdir(directory)
            count = random.randint(1 ,100)
            imsave(str(count)+".jpg", result_1[0])
            cur_dir = r'D:/Videh_Acads/Machine_Learning/GBD_Project/temp/'
            os.chdir(cur_dir)
            pth=os.path.join(directory,str(count)+".jpg")
            pths="./static/predictions\\" + str(count) + ".jpg"
            #img_pred=image.load_img(img_location,target_size=(1084,1084))
            #x = image.img_to_array(img_pred)
            #x=np.expand_dims(x,axis=0)
            # images=np.vstack([x])
            #val=model.predict(x)
            # hu=np.argmax(val, axis =1)
            # st=''
            # if hu==0:
            #     st="Melanocytic nevi"
            # elif hu==1:
            #     st="dermatofibroma"
            # elif hu==2:
            #     st="Benign keratosis-like lesions"
            # elif hu==3:
            #     st="Basal cell carcinoma"
            # elif hu==4:
            #     st="Actinic keratoses"
            # elif hu==5:
            #     st="Vascular lesions"
            # elif hu==6:
            #     st="Dermatofibroma"

            return render_template("index.html", output=pth, img_loc=pths,orig_image=img_location)

if __name__== "__main__":
    app.run(debug=True)
