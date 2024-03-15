import json
import h5py
import numpy as np
import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model
from tensorflow import exp
import losses
import cmsml
import psutil
import plotting

class Trainer:
    #Class to load .json file and launch trainings

    def __init__(self,json_file,layer,axis):
        #axis: "x" or "y"
        assert axis=="x" or axis=="y", f"Invalid axis '{axis}'. Valid choices are 'x' or 'y'"
        self.axis = axis
        self.testing_input_flag = False
        self.training_input_flag = False
        self.layer = layer
        self.resolution = -999.
        self.bias = -999.
        
        with open(json_file) as f:
            config = json.load(f)[layer]
            self.epochs = config["epochs"]
            self.batch_size = config["batch_size"]
            self.batches_per_epoch = config["batches_per_epoch"]
            self.loss_name = config["loss_name"]

            if axis=="x":
                self.train_h5 = config["train_x_h5"]
                self.test_h5 = config["test_x_h5"]
                self.checkpoint = config["checkpoint_x"]
                self.model_dest = config["model_dest_x"]
                self.pitch      = config["pitch_x"]
            else:
                self.train_h5 = config["train_y_h5"]
                self.test_h5 = config["test_y_h5"]
                self.checkpoint = config["checkpoint_y"]
                self.model_dest = config["model_dest_y"]
                self.pitch      = config["pitch_y"]

    def prepare_training_input(self):
        n_clusters = -1#Set to -1 for all or some number like 100000 for faster debugging

        print(f"Loading training clusters from {self.train_h5}")
        f = h5py.File(self.train_h5, 'r')
        pix_flat_train = f[f'{self.axis}_flat'][0:n_clusters]
        cota_train = f['cota'][0:n_clusters]
        cotb_train = f['cotb'][0:n_clusters]
        position_train = f[self.axis][0:n_clusters] 
        clustersize_train = f[f'clustersize_{self.axis}'][0:n_clusters]
        f.close()

        #Random ordering of clusters, is this really needed?
        perm = np.arange(len(pix_flat_train)) 
        np.random.shuffle(perm)
        
        pix_flat_train = pix_flat_train[perm]
        cota_train = cota_train[perm]
        cotb_train = cotb_train[perm]
        position_train = position_train[perm]
        clustersize_train = clustersize_train[perm]
        angles_train = np.hstack((cota_train,cotb_train))

        self.pixels_train = pix_flat_train
        self.position_train = position_train
        self.angles_train = angles_train
        self.clustersize_train = clustersize_train

        self.training_input_flag = True
        print("Using {:.0f} MB of memory".format(psutil.Process().memory_info().rss / 1024 ** 2))

    def prepare_testing_input(self):
        n_clusters = -1#Set to -1 for all or some number like 100000 for faster debugging

        print(f"Loading testing clusters from {self.test_h5}")
        f = h5py.File(self.test_h5, 'r')
        pix_flat_test = f[f'{self.axis}_flat'][0:n_clusters]
        cota_test = f['cota'][0:n_clusters]
        cotb_test = f['cotb'][0:n_clusters]
        position_test = f[self.axis][0:n_clusters] 
        clustersize_test = f[f'clustersize_{self.axis}'][0:n_clusters]
        angles_test = np.hstack((cota_test,cotb_test))
        f.close()

        self.pixels_test = pix_flat_test
        self.position_test = position_test
        self.angles_test = angles_test
        self.clustersize_test = clustersize_test
        print(np.shape(self.pixels_test))
        print(np.shape(self.position_test))
        print(np.shape(self.angles_test))
        print(np.shape(self.clustersize_test))
        print("Using {:.0f} MB of memory".format(psutil.Process().memory_info().rss / 1024 ** 2))
        self.testing_input_flag = True

    def train(self):
        if not self.training_input_flag:
            self.prepare_training_input()
        optimizer = Adam()
        #Maybe make these configurable as well
        validation_split = 0.02
        dropout_level = 0.10

        train_time = time.process_time()

        if self.axis=="x":
            inputs = Input(shape=(13,1)) #13 in x dimension
        else:
            inputs = Input(shape=(21,1)) #21 in y dimension because they are longer on average
        angles = Input(shape=(2,))
        
        #This can and should be optimized
        x = Conv1D(64, kernel_size=3, padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(dropout_level)(x)
        x_cnn = Flatten()(x)
        concat_inputs = concatenate([x_cnn,angles])

        #One branch of network for position estimate
        position = Dense(32)(concat_inputs)
        position = Activation("relu")(position)
        position = Dense(32)(position)
        position = Activation("relu")(position)
        position = BatchNormalization()(position)
        position = Dropout(dropout_level)(position)
        position = Dense(1)(position)

        #Other branch of network for error (variance) estimate
        variance = Dense(32)(concat_inputs)
        variance = Activation("relu")(variance)
        variance = Dense(32)(variance)
        variance = Activation("relu")(variance)
        variance = BatchNormalization()(variance)
        variance = Dropout(dropout_level)(variance)
        
        #variance = Dense(1, activation = "softplus")(variance)#We want something positive, but not capped to [0,1] like sigmoid
        #Alternative to using softplus, we can transform possibly negative output to something positive
        variance = Dense(1, activation = "relu")(variance)
        variance = exp(variance)

        position_variance = concatenate([position, variance])

        model = Model(inputs=[inputs,angles],outputs=[position_variance])

        # Display a model summary
        model.summary()
        #from keras.utils.vis_utils import plot_model
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        #If we want to continue from a saved training
        #history = model.load_weights("checkpoints/cp_x%s.ckpt"%(img_ext))

        # Compile the model
        run_eagerly = False #Set to true if we want to printout inputs during training, useful for debugging
        loss_func   = getattr(losses,self.loss_name)
        model.compile(loss=loss_func,optimizer=optimizer,run_eagerly=run_eagerly,metrics = [losses.mse_position,losses.mean_pulls,loss_func])

        #Load weights from checkpoint if exists
        #checkpoint_filepath=self.checkpoint
        checkpoint_filepath=self.checkpoint
        if os.path.exists(checkpoint_filepath+".index"):
            print(f"Loading weights from {checkpoint_filepath}")
            model.load_weights(checkpoint_filepath)
            
        #Uncomment if you want to save weights in a different file!
        #checkpoint_filepath = checkpoint_filepath.replace(".ckpt",".{epoch:02d}-{val_loss:.2f}.ckpt")

        callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
                        save_best_only=True,
                        save_weights_only=True,
                        #monitor='val_loss',
                        monitor=self.loss_name,
                        save_freq="epoch")
                        #save_freq=1000)
        ]

        # Fit data to model
        history = model.fit([self.pixels_train[:,:,np.newaxis],self.angles_train], [self.position_train],
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        steps_per_epoch=self.batches_per_epoch)


        model.save(self.model_dest)
        cmsml.tensorflow.save_graph(self.model_dest+".pb", model, variables_to_constants=True)
        cmsml.tensorflow.save_graph(self.model_dest+".pb.txt", model, variables_to_constants=True)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        history_plot = f"plots/{self.layer}_{self.axis}_history.png"
        plotting.plot_dnn_loss(history.history,history_plot)

        print("Training time: {:.0f}s".format(time.process_time()-train_time))

    def test(self):
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})

        start = time.process_time()
        pred = model.predict([self.pixels_test[:,:,np.newaxis],self.angles_test], batch_size=400000)
        inference_time_x = time.process_time() - start
        print("Inference on {} cluster took {:.3f} s".format(len(pred),inference_time_x))
        residuals = pred[:,0] - self.position_test[:,0]
        pulls     = residuals/np.sqrt(pred[:,1])

        self.resolution = np.std(residuals)
        self.bias       = np.mean(residuals)
        print("Residuals mean and std (microns): {:.3f} +/- {:.3f}".format(self.bias,self.resolution))
        print("Pulls mean and std: {:.3f} +/- {:.3f}".format(np.mean(pulls),np.std(pulls)))

        residuals_output_file = f"plots/{self.layer}_{self.axis}_residuals.pdf"
        pulls_output_file = f"plots/{self.layer}_{self.axis}_pulls.pdf"
        plot_name = f"{self.layer}_{self.axis}"
        plotting.plot_residuals(residuals,residuals_output_file,plot_type="Residuals",name=plot_name)
        plotting.plot_residuals(pulls,pulls_output_file,plot_type="Pulls",name=plot_name)

    def visualize(self):
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})
        n_to_plot = 10
        clusters_for_plotting = self.pixels_test[:n_to_plot]
        angles_for_plotting   = self.angles_test[:n_to_plot]
        position_for_plotting = self.position_test[:n_to_plot,0]
        pred = model.predict([clusters_for_plotting[:,:,np.newaxis],angles_for_plotting])
        
        plotting_data_sets = []
        plotting_file_name = f"plots/{self.layer}_{self.axis}.pdf"
        for i in range(n_to_plot):
            temp_data_set = {
            'cluster': clusters_for_plotting[i],
            'angles': angles_for_plotting[i],
            'prediction_variance': pred[i],
            'position': position_for_plotting[i],
            'pixel_pitch': self.pitch,
            'resolution': self.resolution,
            'bias': self.bias

            }
            plotting_data_sets.append(temp_data_set)
        
    
        plotting.plot_clusters(plotting_data_sets,plotting_file_name)