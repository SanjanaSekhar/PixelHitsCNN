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

h5_date = "020522"
h5_ext = "p1_2024_by25k_irrad_BPIXL1"
img_ext = "1dcnn_%s_111022"%h5_ext

# Load data
f = h5py.File('h5_files/train_x_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_train = f['train_x_flat'][...]
#ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
#y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
#clustersize_y_train = f['clustersize_y'][...]

f.close()
#print(inputs_x_train[0])

perm = np.arange(len(xpix_flat_train)) 
np.random.shuffle(perm)
xpix_flat_train = xpix_flat_train[perm]
cota_train = cota_train[perm]
cotb_train = cotb_train[perm]
x_train = x_train[perm]
clustersize_x_train = clustersize_x_train[perm]
print(xpix_flat_train.shape, cota_train.shape, cotb_train.shape)
inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
#inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))

#h5_date = "110121"
#h5_ext = "p1_2024_irrad_BPIXL1"

f = h5py.File('h5_files/test_x_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_test = f['test_x_flat'][...]
#ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...] 
#y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
#clustersize_y_test = f['clustersize_y'][...]
inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
#inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
angles_test = np.hstack((cota_test,cotb_test))
f.close()
'''
h5_date = "082821"
h5_ext = "p1_2018_irrad_BPIXL1_file2"

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_train = np.vstack((xpix_flat_train,f['train_x_flat'][...]))
ypix_flat_train = np.vstack((ypix_flat_train,f['train_y_flat'][...]))
cota_train = np.vstack((cota_train,f['cota'][...]))
cotb_train = np.vstack((cotb_train,f['cotb'][...]))
x_train = np.vstack((x_train,f['x'][...])) 
y_train = np.vstack((y_train,f['y'][...]))
clustersize_x_train = np.vstack((clustersize_x_train,f['clustersize_x'][...]))
clustersize_y_train = np.vstack((clustersize_y_train,f['clustersize_y'][...]))

inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))
f.close()
'''
#print(angles_train.shape)
#print(xpix_flat_test[:30])
#print(ypix_flat_test[:30])
#print("clustersize of 1: ",len(np.argwhere(clustersize_x_train==1)))
#for i in range(50):
#	if clustersize_x_train[i]==1:
#	print(xpix_flat_train[i])
print(np.amin(cota_train),np.amax(cota_train))
print(np.amin(cota_test),np.amax(cota_test))
plot_cot(cota_train,"cota_train",img_ext)
plot_cot(cota_test,"cota_test",img_ext)	
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
n_epochs_x = 30
n_epochs_y = 40
optimizer = Adam(lr=0.001)
validation_split = 0.3

train_time_cota = time.clock()
#train flat x

inputs = Input(shape=(13,1)) #13 in x dimension + 2 angles
#angles = Input(shape=(2,))
x = Conv1D(64, kernel_size=3, padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2,padding='same')(x)
x = Conv1D(128, kernel_size=3, padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2,padding='same')(x)
x = Dropout(0.25)(x)
x = Conv1D(64, kernel_size=3, padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2,padding='same')(x)
# x = Conv1D(64, kernel_size=2, padding="same")(x)
# x = Activation("relu")(x) 
# x = BatchNormalization(axis=-1)(x)
# x = MaxPooling1D(pool_size=2,padding='same')(x)
#x = Dropout(0.25)(x)

x_cnn = Flatten()(x)
#concat_inputs = concatenate([x_cnn,angles])
x = Dense(32)(x_cnn)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(32)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(1)(x)
cota = Activation("linear", name="cota")(x)


model = Model(inputs=[inputs],
              outputs=[cota]
              )

# Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_cota%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse']
              )

callbacks = [
EarlyStopping(patience=5),
ModelCheckpoint(filepath="checkpoints/cp_cota%s.ckpt"%(img_ext),
                save_best_only=True,
		            save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([xpix_flat_train[:,:,np.newaxis]], [cota_train],
                batch_size=batch_size,
                epochs=n_epochs_x,
                callbacks=callbacks,
                validation_split=validation_split)

cmsml.tensorflow.save_graph("data/graph_cota_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_cota_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

#plot_dnn_loss(history.history,'x',img_ext)

print("cota training time for dnn",time.clock()-train_time_cota)

start = time.clock()
cota_pred = model.predict([xpix_flat_test[:,:,np.newaxis]], batch_size=9000)
inference_time_cota = time.clock() - start
residuals_cota = cota_pred - cota_test
RMS_cota = np.std(residuals_cota)
print(np.amin(residuals_cota),np.amax(residuals_cota))
print("RMS_cota = %f\n"%(RMS_cota))


plot_residuals(residuals_cota,'1dcnn','cota',img_ext)
#plot_by_clustersize(residuals_x,clustersize_x_test,'x',img_ext)


