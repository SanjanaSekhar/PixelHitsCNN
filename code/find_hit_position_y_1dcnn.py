#============================
# Author: Sanjana Sekhar
# Date: 17 Jan 21
#============================
'''
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
'''
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score
import numpy as np
import time
from plotter import *
from tensorflow.keras.callbacks import EarlyStopping
import cmsml

h5_date = "082521"
h5_ext = "p1_2018_irrad_BPIXL1"
img_ext = "1dcnn_%s_aug17"%h5_ext

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_train = f['train_x_flat'][...]
ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
clustersize_y_train = f['clustersize_y'][...]
inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))
f.close()
#print(inputs_x_train[0])

f = h5py.File('h5_files/test_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_test = f['test_x_flat'][...]
ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...] 
y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
clustersize_y_test = f['clustersize_y'][...]
inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
angles_test = np.hstack((cota_test,cotb_test))
f.close()
print(angles_train.shape)
print(xpix_flat_test[:30])
print(ypix_flat_test[:30])
print("clustersize of 1: ",len(np.argwhere(clustersize_y_train==1)))
'''
norm_x = np.amax(xpix_flat_train)
xpix_flat_train/=norm_x
xpix_flat_test/=norm_x
norm_y = np.amax(ypix_flat_train)
ypix_flat_train/=norm_y
ypix_flat_test/=norm_y


test_c = np.zeros((13,21))
test_c[6,9]=4425
test_c[6,10]=14403
test_ang = np.array([-0.321207,-0.348136]).reshape((1,2))
print(test_ang.shape)
test_cx = test_c.sum(axis=1).reshape((1,13))
print(test_cx.shape)
test_cy = test_c.sum(axis=0).reshape((1,21))
'''
# Model configuration
batch_size = 512
loss_function = 'mse'
n_epochs_x = 20
n_epochs_y = 20
optimizer = Adam(lr=0.0001)
validation_split = 0.2


train_time_y = time.clock()

#train flat y

inputs = Input(shape=(21,1)) #13 in y dimension + 2 angles
angles = Input(shape=(2,))
y = Conv1D(64, kernel_size=3, padding="same",kernel_regularizer='l2')(inputs)
y = Activation("relu")(y)
y = Conv1D(64, kernel_size=3, padding="same",kernel_regularizer='l2')(y)
y = Activation("relu")(y)
y = BatchNormalization(axis=-1)(y)
y = MaxPooling1D(pool_size=2,padding='same')(y)
y = Dropout(0.25)(y)
#y = Conv1D(32, kernel_size=3, padding="same",kernel_regularizer='l2')(y)
#y = Activation("relu")(y)
#y = Conv1D(16, kernel_size=3, padding="same",kernel_regularizer='l2')(y)
#y = Activation("relu")(y) 
#y = BatchNormalization(axis=-1)(y)
#y = MaxPooling1D(pool_size=2,padding='same')(y)
#y = Dropout(0.25)(y)

y_cnn = Flatten()(y)
concat_inputs = concatenate([y_cnn,angles])
y = Dense(64,kernel_regularizer='l2')(concat_inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(128,kernel_regularizer='l2')(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(64,kernel_regularizer='l2')(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(1)(y)
y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs,angles],
              outputs=[y_position]
              )

# Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_y%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse']
              )



callbacks = [
#EarlyStopping(patience=5),
ModelCheckpoint(filepath="checkpoints/cp_y%s.ckpt"%(img_ext),
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([ypix_flat_train[:,:,np.newaxis],angles_train], [y_train],
                batch_size=batch_size,
                epochs=n_epochs_y,
                validation_split=validation_split,
                callbacks=callbacks)

cmsml.tensorflow.save_graph("data/graph_y_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_y_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

plot_dnn_loss(history.history,'y',img_ext)

print("y training time for dnn",time.clock()-train_time_y)

start = time.clock()
y_pred = model.predict([ypix_flat_test[:,:,np.newaxis],angles_test], batch_size=9000)
inference_time_y = time.clock() - start

#print("inference_time for dnn= ",(inference_time_x+inference_time_y))


residuals_y = y_pred - y_test
RMS_y = np.sqrt(np.mean(residuals_y*residuals_y))
print(np.amin(residuals_y),np.amax(residuals_y))
print("RMS_y = %f\n"%(RMS_y))


mean_y, sigma_y = norm.fit(residuals_y)
print("mean_y = %0.2f, sigma_y = %0.2f"%(mean_y,sigma_y))

plot_residuals(residuals_y,mean_y,sigma_y,RMS_y,'y',img_ext)

plot_by_clustersize(residuals_y,clustersize_y_test,'y',img_ext)

