import pandas as pd
import numpy as np
import csv
import sys
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.models import load_model
from keras.losses import huber_loss
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import Conv2D, MaxPool2D, Dropout, Input, UpSampling2D, Concatenate
from keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.io_utils import HDF5Matrix
from keras import metrics

def huber_loss_wrapper(**huber_loss_kwargs):
    def huber_loss_wrapped_function(y_true, y_pred):
        return huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return huber_loss_wrapped_function



def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    #c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    p=MaxPool2D((2,2),(2,2))(c)
    #p=Dropout(rate=0.1)(p)
    return c,p

def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    us=UpSampling2D((2,2))(x)
    concat=Concatenate()([us,skip])
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):
    c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    c=BatchNormalization()(c)
    #c=LeakyReLU(alpha=0.1)(c)
    #c=Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    #c=LeakyReLU(alpha=0.1)(c)
    return c

def UNet():
    #F=[128,256,512,1024]
    #F=[32,64,128,256]
    F=[32,64,128,256]
    #F=[64,128,256,512]
    ginputs=Input((800,800,36))
    iinputs=Input((200,200,6))
    finputs=Input((100,100,10))
    #inputs=keras.layers.Input((40,40,14))
    print('F=',F, flush=True)
    p0=ginputs
    c1, g1 = down_block(p0,F[0]) #800->400
    c1, g2 = down_block(g1,F[1]) #400->200
    #c1, g3 = down_block(g2,F[2]) #200->100

    p1=iinputs
    con1=Concatenate()([g2,p1])
    c2, pi1= down_block(con1,F[3]) #200->100

    #p2=finputs
    #con2=Concatenate()([pi1,p2])	
    c3, pi2= down_block(pi1,F[3]) #100->50
    c4, pi3= down_block(pi2,F[3]) #50->25

    bn=bottleneck(pi3,F[3])

    u1=up_block(bn, c4, F[2]) #25-50
    u2=up_block(u1, c3, F[1]) #50-100
    u3=up_block(u2, c2, F[0]) #100-200

    outputs=Conv2D(2,(1,1), padding="same", activation="sigmoid")(u3)
    model=Model([ginputs,iinputs], outputs)
    return model

path='/scratch/x1879a02/geoDIP/model/unet_withgoes/input/3hr/data2/'
label=HDF5Matrix(path+'tocate_label_2019_200_3h_bi_01.h5', 'data')
#label=HDF5Matrix(path+'label_2017_120_unet.h5', 'data')
imerg=HDF5Matrix(path+'imerg_2019_200_3h.h5','data')
goes=HDF5Matrix(path+'nor_goes_2019_800_3h_re.h5','data')
#gfs=HDF5Matrix(path+'nor_gfs_2019_100_3h.h5','data')



print(imerg.shape, label.shape,goes.shape, flush=True)


model=UNet()
print(model.summary(),flush=True)
model = multi_gpu_model(model,gpus=2)

metrics = [
    metrics.FalseNegatives(name="fn"),
    metrics.FalsePositives(name="fp"),
    metrics.TrueNegatives(name="tn"),
    metrics.TruePositives(name="tp"),
    metrics.Precision(name="precision"),
    metrics.Recall(name="recall"),
]


model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=metrics)

epochs=500
batch_size=16
earlystopper = EarlyStopping(patience=50,verbose=1, monitor='val_loss')
checkpointer = ModelCheckpoint('model_ck_mse_new.h5', save_best_only=True, verbose=1)

history=model.fit([goes,imerg], label, epochs=epochs, batch_size=batch_size,
          validation_split=0.3, callbacks=[earlystopper,checkpointer], verbose=2) 

results=pd.DataFrame(history.history)
print(results, flush=True)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    results.to_csv(f)



