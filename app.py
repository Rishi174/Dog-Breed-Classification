import os
import warnings
import numpy as np
import keras
from keras.layers import Add,Input,Conv2D, Flatten, Dense, Activation,MaxPooling2D, BatchNormalization, Dropout, AveragePooling2D
from keras.initializers import glorot_uniform, he_normal
from keras.regularizers import l2
from keras.models import Model
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
warnings.filterwarnings("ignore")


app = Flask(__name__)
breeds = {0:'beagle',1:'chihuahua',2:'doberman',3:'french_bulldog',4:'golden_retriever',5:'malamute',6:'pug',
          7:'saint_bernard',8:'scottish_deerhound', 9:'tibetan_mastiff'}
def convolutional_block(input_feature_map, filter_sizes, stride=2, regularizer = None):
    
    filters_1, filters_2, filters_3 = filter_sizes
    X = input_feature_map
    
    X_shortcut = X

    X = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(stride, stride), padding='valid', 
               kernel_initializer=he_normal(seed=0))(X)
    
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters_2, kernel_size=(3, 3), strides=(1, 1), padding='same',  
               kernel_initializer=he_normal(seed=0), kernel_regularizer = regularizer)(X)
    
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=he_normal(seed=0))(X)
    
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(stride, stride), padding='valid', 
                        kernel_initializer=he_normal(seed=0))(X_shortcut)
    
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block(input_feature_map, filter_sizes,regularizer = None):
   
    filters_1, filters_2, filters_3 = filter_sizes
    
    X = input_feature_map
    X_shortcut = X
   
    X = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=he_normal(seed=0))(X)
    
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters_2, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_initializer=he_normal(seed=0), kernel_regularizer = regularizer)(X)
    
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               kernel_initializer=he_normal(seed=0))(X)
    
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

def create_model():
    input_layer = Input(shape=(224,224,3))
    
    X = Conv2D(64, (7, 7), strides=(2, 2), padding = 'same',kernel_initializer=he_normal(seed=0))(input_layer)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = convolutional_block(X,filter_sizes=[64, 64, 256], stride=1)
    X = identity_block(X,filter_sizes=[64, 64, 256])
    X = identity_block(X,filter_sizes= [64, 64, 256])
    
    
    X = convolutional_block(X,filter_sizes=[128, 128, 512],stride=2, regularizer = l2(0.01))
    X = identity_block(X,filter_sizes= [128, 128, 512])
    X = identity_block(X,filter_sizes= [128, 128, 512], regularizer = l2(0.01))
    X = identity_block(X,filter_sizes= [128, 128, 512])
    
    X = convolutional_block(X,filter_sizes=[256, 256, 1024], stride=2, regularizer = l2(0.01))
    X = identity_block(X,filter_sizes= [256, 256, 1024])
    X = identity_block(X,filter_sizes= [256, 256, 1024], regularizer = l2(0.01))
    X = identity_block(X,filter_sizes= [256, 256, 1024])
    X = identity_block(X,filter_sizes= [256, 256, 1024], regularizer = l2(0.01))
    X = identity_block(X,filter_sizes=[256, 256, 1024])
    
    X = X = convolutional_block(X,filter_sizes=[512, 512, 2048], stride=2, regularizer = l2(0.01))
    X = identity_block(X,filter_sizes=[512, 512, 2048])
    X = identity_block(X,filter_sizes=[512, 512, 2048],regularizer = l2(0.01))
    
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    X = Flatten()(X)
    X=Dense(1024, activation='relu',kernel_initializer=he_normal(seed=0))(X)
    X=Dropout(0.4)(X)
    X=Dense(512, activation='relu',kernel_initializer=he_normal(seed=0))(X)
    X=Dropout(0.3)(X)
    X=Dense(256, activation='relu',kernel_initializer=he_normal(seed=0))(X)
    X=Dropout(0.2)(X)
    X=Dense(128, activation='relu',kernel_initializer=he_normal(seed=0))(X)
    output = Dense(10,activation='softmax',kernel_initializer=glorot_uniform(seed=0))(X)
    
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001),loss='categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

#In order to load the trained weights, we need to re-create the architecture to load the weights. 
model = create_model()
model.load_weights('weights.h5')

def model_predict(img_path,model):
    img = Image.open(img_path)
    img = img.resize((224,224))
    arr = np.array(img)
    arr = arr/255.0
    arr = np.expand_dims(arr,axis=0)
    preds = model.predict(arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
        
       
    f = request.files['file'] #The submit button in the form has an id which can be used here to capture the image path
       
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath,'files',secure_filename(f.filename))
    f.save(file_path)
        
        
    preds = model_predict(file_path,model)

    pred_class = np.argmax(preds)
    prob = preds[0][pred_class]
    rounded_prob = round(prob,2)
    result = f'Breed : {breeds[pred_class]} Score: {round(rounded_prob,2)}'
    return result



if __name__ == '__main__':
    app.run(debug=True)


