#!/usr/bin/python

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,merge
import keras.metrics
from keras.models import Model
from keras.utils import np_utils
import random
import numpy as np
import PIL
import tensorflow as tf

# Don't modify BEGIN

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.15
sess = tf.Session(config=config)

seed = 7
np.random.seed(seed)

# END

WIDTH = 160
HEIGHT = 60
CHANNEL = 3

def one_hot_encode (label) :
    return np_utils.to_categorical(np.int32(list(label)), 10)

def accuracy(test_labels, predict_labels):
    y1 = K.cast(K.equal(K.argmax(test_labels[:,0,:]), K.argmax(predict_labels[:,0,:])), K.floatx())
    y2 = K.cast(K.equal(K.argmax(test_labels[:,1,:]), K.argmax(predict_labels[:,1,:])), K.floatx())
    y3 = K.cast(K.equal(K.argmax(test_labels[:,2,:]), K.argmax(predict_labels[:,2,:])), K.floatx())
    y4 = K.cast(K.equal(K.argmax(test_labels[:,3,:]), K.argmax(predict_labels[:,3,:])), K.floatx())
    acc = K.mean(y1 * y2 * y3 * y4)
    return acc

def load_data(path,train_ratio):
    datas = []
    labels = []
    input_file = open(path + 'labels.txt')
    for i,line in enumerate(input_file):
        chal_img = PIL.Image.open(path + str(i) + ".png").resize((WIDTH, HEIGHT))
        data = np.array(chal_img).astype(np.float32)
        data = np.multiply(data, 1/255.0)
        data = np.asarray(data)
        datas.append(data)
        labels.append(one_hot_encode(line.strip()))
    input_file.close()
    datas_labels = zip(datas,labels)
    random.shuffle(datas_labels)
    (datas,labels) = zip(*datas_labels)
    size = len(labels)
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[ 0 : train_size ])
    test_datas = np.stack(datas[ train_size : size ])
    train_labels = np.stack(labels[ 0 : train_size ])
    test_labels = np.stack(labels[ train_size : size])
    return (train_datas,train_labels,test_datas,test_labels)

def get_cnn_net():
    ### what I have done ##########################################################
    # 1) increase number of filters to 64, 128 for second and third convolution   #
    # 2) add a conv256+pool+drop                                                  #
    # 3) add dense256 layer                                                       #
    ###############################################################################
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    x = Conv2D(32, (5, 5), padding='valid', input_shape=(HEIGHT, WIDTH, CHANNEL), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
#   TODO!!!
#   add a Conv Layer
    x = Conv2D(64, (5, 5), activation='relu')(x)
#   add a Pooling Layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.15)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.15)(x)
    
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.15)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x1 = Dense(10, activation='softmax')(x)
    x2 = Dense(10, activation='softmax')(x)
    x3 = Dense(10, activation='softmax')(x)
    x4 = Dense(10, activation='softmax')(x)
    x = concatenate([x1,x2,x3,x4])
    x = Reshape((4,10))(x)
    model = Model(inputs=inputs, outputs=x)
    
    model.compile(loss='categorical_crossentropy', loss_weights=[1.], optimizer='adam', metrics=[accuracy])
    return model

# Don't modify BEGIN

(train_datas,train_labels,test_datas,test_labels) = load_data('/home/share/captcha_data_dontcp/',0.9)
model = get_cnn_net()
print model
model.fit(train_datas, train_labels, epochs=32, batch_size=32, verbose=1, validation_split=0.1)
predict_labels = model.predict(test_datas,batch_size=32)
test_size = len(test_labels)
y1 = test_labels[:,0,:].argmax(1) == predict_labels[:,0,:].argmax(1)
y2 = test_labels[:,1,:].argmax(1) == predict_labels[:,1,:].argmax(1)
y3 = test_labels[:,2,:].argmax(1) == predict_labels[:,2,:].argmax(1)
y4 = test_labels[:,3,:].argmax(1) == predict_labels[:,3,:].argmax(1)
acc = (y1 * y2 * y3 * y4).sum() * 1.0

print '\nmodel evaluate:\nacc:', acc/test_size
print 'y1',(y1.sum()) *1.0/test_size
print 'y2',(y2.sum()) *1.0/test_size
print 'y3',(y3.sum()) *1.0/test_size
print 'y4',(y4.sum()) *1.0/test_size

K.clear_session()
del sess

# END
