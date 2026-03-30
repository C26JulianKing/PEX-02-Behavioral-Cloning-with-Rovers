import data_gen
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
import tensorflow.keras.backend as K

# Configuration parameters
DEVICE = "/GPU:0" 
DATA_PATH = "processed/bw"  
MODEL_NUM = 6 
TRAINING_VER = 3
NUM_EPOCHS = 50  
BATCH_SIZE = 32  
TRAIN_VAL_SPLIT = 0.8  

# Custom R2Score for Keras 2.13 Compatibility
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.sum_squares_residual = self.add_weight(name='ss_res', initializer='zeros')
        self.sum_squares_total = self.add_weight(name='ss_tot', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.mean = self.add_weight(name='mean', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Residual sum of squares
        residual = tf.square(y_true - y_pred)
        self.sum_squares_residual.assign_add(tf.reduce_sum(residual))
        
        # Total sum of squares (requires tracking mean)
        new_count = self.count + tf.cast(tf.shape(y_true)[0], tf.float32)
        delta = tf.reduce_mean(y_true) - self.mean
        self.mean.assign_add(delta * tf.cast(tf.shape(y_true)[0], tf.float32) / new_count)
        self.sum_squares_total.assign_add(tf.reduce_sum(tf.square(y_true - self.mean)))
        self.count.assign(new_count)

    def result(self):
        return 1 - (self.sum_squares_residual / (self.sum_squares_total + K.epsilon()))

    def reset_state(self):
        self.sum_squares_residual.assign(0.0)
        self.sum_squares_total.assign(0.0)
        self.count.assign(0.0)
        self.mean.assign(0.0)

# Define the CNN model structure
def define_model(input_shape=(160, 120, 1)):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(),

        Flatten(),

        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2)
    ])

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mse', metrics=['mae', R2Score()])
    return model

def train_model(amt_data=1.0):
    samples = data_gen.get_sequence_samples(DATA_PATH, sequence_size=13)
    
    if amt_data < 1.0:
        samples, _ = data_gen.split_samples(samples, fraction=amt_data)

    train_samples, val_samples = data_gen.split_samples(samples, fraction=TRAIN_VAL_SPLIT)
    
    train_steps = int(len(train_samples) / BATCH_SIZE)
    val_steps = int(len(val_samples) / BATCH_SIZE)

    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE)
    
    # Keras 2.13 handles device placement via the tf.device context manager
    with tf.device(DEVICE):
        model = define_model(input_shape=(160, 120, 1))
        model.summary()
        
        filePath = "models/rover_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # fit_generator was merged into fit in later Keras 2.x versions
        history = model.fit(train_gen, 
                            epochs=NUM_EPOCHS, 
                            steps_per_epoch=train_steps,
                            validation_data=val_gen, 
                            validation_steps=val_steps,
                            callbacks=[checkpoint_best])

    return history

def summarize_diagnostics(histories):
    for i in range(len(histories)):
        pyplot.subplot(len(histories), 1, 1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
    pyplot.show()

def main():
    history = train_model(1.0)
    summarize_diagnostics([history])

if __name__ == "__main__":
    main()
