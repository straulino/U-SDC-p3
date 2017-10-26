import os
import random
import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import cv2

import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/', "location of the data")
tf.app.flags.DEFINE_string('model_file','model.h5', "model for driving")
tf.app.flags.DEFINE_float('angle', 0.0225, "side cameras' correction") 
tf.app.flags.DEFINE_integer('epochs', 10, "number of training epochs")
tf.app.flags.DEFINE_integer('raw_batch_size', 64, "The size of the batches") 
tf.app.flags.DEFINE_bool('reset', True, "True will start from scratch.")

def get_data():
    print("Loading data in directory {}...".format(FLAGS.data_dir))
    data = []
    with open(os.path.join(FLAGS.data_dir, "driving_log.csv"), 'r') as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            data.append(row)
    return data

def balance_data(data):
    v_steering = []
    print("Data size")
    print(len(data))
    for line in data:
        v_steering.append(float(line[3]))    

    v_steering = np.array(v_steering)
    num_bins = 100

    avg_samples_per_bin = len(v_steering)/num_bins
    hist, bins = np.histogram(v_steering, num_bins)
    #width = 0.7 * (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)  
    #plt.savefig('before.png')  
    #plt.show()
    
    prob_v = []
    target = avg_samples_per_bin*1.5
    for i in range(num_bins):
        if hist[i] < target:
            prob_v.append(1.)
        else:
            prob_v.append(1./(hist[i]/target))

    keep_list = []

    for i in range(len(v_steering)):
        for j in range(num_bins):
            if v_steering[i] > bins[j] and v_steering[i] <= bins[j+1]:                
                if np.random.rand() < prob_v[j]:
                    keep_list.append(i)

    data_balanced = []

    for i in range(len(keep_list)):
        data_balanced.append(data[keep_list[i]])

    v_steering = []
    print("Data size")
    print(len(data_balanced))
    for line in data_balanced:
        v_steering.append(float(line[3]))    

    v_steering = np.array(v_steering)
    #num_bins = 100

    #avg_samples_per_bin = len(v_steering)/num_bins
    #hist, bins = np.histogram(v_steering, num_bins)
    #width = 0.7 * (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)
    #plt.savefig('after.png')     
    #plt.show()

    return data_balanced


def augment(image):
    case = np.random.randint(3)

    # hsv filter
    if case == 0: 
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)        
        hsv = (np.random.randint(low=0.5, high=1.5) * hsv).round()
        image = np.minimum.reduce([hsv, 255*np.ones_like(hsv)])             
        
    # add shadow
    if case == 1:
        h, w = image.shape[0], image.shape[1]
        [x1, x2] = np.random.choice(w, 2, replace=False)
        k = h / (x2 - x1)
        b = - k * x1
        for i in range(h):
            c = int((i - b) / k)
            image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)

    return image

def generator(data):
    while True:
        random.shuffle(data)
        for first in range(0, len(data), FLAGS.raw_batch_size):
            last = min(first + FLAGS.raw_batch_size, len(data))
            center_images = []
            left_images = []
            right_images = []
            steer = []
            for c, l, r, steering, throttle, brake, speed in data[first:last]:
                # following the advice given on the Slack channel
                img_c = cv2.imread(os.path.join(FLAGS.data_dir, c.strip()))
                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                img_l = cv2.imread(os.path.join(FLAGS.data_dir, l.strip()))
                img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
                img_r = cv2.imread(os.path.join(FLAGS.data_dir, r.strip()))
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
				# each of the images is randomly augmented
                center_images.append(augment(img_c))
                left_images.append(augment(img_l))
                right_images.append(augment(img_r))
                steer.append(float(steering))            
            steer = np.array(steer)
            X = np.concatenate((np.array(center_images), np.array(left_images), np.array(right_images)))
            y = np.concatenate((steer, steer + FLAGS.angle, steer - FLAGS.angle))
            
            X = np.concatenate((X, np.fliplr(X)))
            y = np.concatenate((y, -y))
            
            yield X, y


def get_model():    
    model = Sequential()
    model.add(Cropping2D(
        cropping=((60, 30), (0, 0)), 
        input_shape=(160, 320, 3)
    )) 
    model.add(Lambda(lambda img: tf.image.resize_images(img, (35, 160))))
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    model.add(Conv2D(filters=32, kernel_size=5, strides=(1,2), activation='relu', padding="valid"))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2), activation='relu', padding="valid"))    
    model.add(Conv2D(filters=48, kernel_size=3, strides=(2,2), activation='relu', padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu', padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=1, strides=(2,2), activation='relu', padding="valid"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(786, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))    
    model.add(Dense(1))
    return model


def main(_):
    data = get_data()
    data = balance_data(data)

    random.shuffle(data)
    training_data = data[:int(0.8*len(data))]
    validation_data = data[int(0.8*len(data)):]
    train_gen = generator(training_data)
    valid_gen = generator(validation_data)
    model = get_model()
    if not FLAGS.reset:
        model = load_model(FLAGS.model_file)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(len(training_data)/FLAGS.raw_batch_size)),
        epochs=FLAGS.epochs,
        callbacks=[
            ModelCheckpoint(FLAGS.model_file),
            TensorBoard(
                write_images=True,
                batch_size=FLAGS.raw_batch_size
            )
        ],
        verbose=1,
        validation_data=valid_gen,
        validation_steps=int(np.ceil(len(validation_data)/FLAGS.raw_batch_size))
    )

if __name__ == '__main__':
    tf.app.run()
