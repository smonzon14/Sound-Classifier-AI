# Author: Sebastian Monzon

from multiclass import MulticlassNSynth
import tensorflow as tf
import numpy as np
import loader
import datetime 

# log file directory
log_dir = "C:\\Users\\smonz\\Documents\\Honors Thesis Project\\GAN\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir) # type: ignore

# Define constants
SHUFFLE_BUFFER_SIZE = 300000 # Just greater than train dataset size
BATCH_SIZE = 64
EXCLUDE_CLASSES = [] # classes to leave of of training
NUM_CLASSES = 11 - len(EXCLUDE_CLASSES)

# load dataset partitions
dataset_train = loader.nsynth_dataset(batch_size=BATCH_SIZE, exclude_classes=EXCLUDE_CLASSES)
dataset_valid = loader.nsynth_dataset(batch_size=BATCH_SIZE, split="valid", exclude_classes=EXCLUDE_CLASSES)
dataset_test = loader.nsynth_dataset(batch_size=BATCH_SIZE, split="test", exclude_classes=EXCLUDE_CLASSES)

# Define the model
model = MulticlassNSynth(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)

# training variables
epochs = 100
learning_rate = 0.007
rlronp=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.4,
                                             patience=2, verbose=1, mode="auto")
estop=tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=4,
                                        verbose=1,mode="auto",    
                                        restore_best_weights=True)

# Compile the model
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the confusion matrix callback
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        # Compute the confusion matrix
        with file_writer.as_default():
            for batch in self.test_data:
                x, labels = batch
                y_pred = np.argmax(model.predict(x, batch_size=BATCH_SIZE), axis=1)
                cm = tf.math.confusion_matrix(labels, y_pred, num_classes=NUM_CLASSES)

                # Reshape the matrix for display in TensorBoard
                cm_image = tf.reshape(tf.cast(cm, tf.float32), (1, NUM_CLASSES, NUM_CLASSES, 1))

                # Write the confusion matrix to TensorBoard
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)
                break

# callbacks
checkpoint_path = "C:\\Users\\smonz\\Documents\\Honors Thesis Project\\GAN\\model_checkpoint"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
cm_callback = ConfusionMatrixCallback(dataset_test)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# print model summary
model.summary()

# train the model
model.fit(dataset_train, 
          epochs=epochs, 
          callbacks=[cp_callback, tensorboard_callback, rlronp, estop, cm_callback], 
          use_multiprocessing=True, 
          validation_data=dataset_valid, 
          )