import tensorflow as tf
import numpy as np
from mpose import MPOSE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


labels = { # Your classes here
          }



def load_mpose(dataset, split, verbose=False, legacy=False):
    
    if legacy:
        return load_dataset_legacy(data_folder='E:/AcT/')
    
    d = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess='random_flip', 
                    velocities=True, 
                    remove_zip=False)
    
    if 'legacy' not in dataset:
        d.reduce_keypoints() #reduce keypoints from 25 to 19
        d.scale_and_center() #preprocessing step
        d.remove_confidence() #preprocessing step
        d.flatten_features() #preprocessing step
        #d.reduce_labels()
        return d.get_data()
    
    elif 'openpose' in dataset:
        X_train, y_train, X_test, y_test = d.get_data()
        print("the shape")
        print(X_train.shape)
        return X_train, transform_labels(y_train), X_test, transform_labels(y_test)
        #return X_train, (y_train), X_test, (y_test)
    else:
        return d.get_data()
        

def random_flip(x, y):
    time_steps = x.shape[0]
    print(time_steps)
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features//2, 2))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features//3, 3))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0,1.0])
    x = tf.reshape(x, (time_steps,-1))
    return x, y


def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.03, dtype=tf.float64)
    x = x + noise
    return x, y


def one_hot(x, y, n_classes):
    #y = int(y)
    return x, tf.one_hot(y, n_classes)


def kinetics_generator(X, y, batch_size):
    while True:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X  = X[ind_list,...]
        y = y[ind_list]
        for count in range(len(y)):
            yield (X[count], y[count])
        
        
def callable_gen(_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen
    
    
def transform_labels(y):
    y_new = []
    for i in y:
        y_new.append(labels[i])
    return np.array(y_new)


def load_dataset_legacy(data_folder, verbose=True):
    X_train = np.load(data_folder + 'X_train.npy')
    y_train = np.load(data_folder + 'Y_train.npy', allow_pickle=True)
    y_train = transform_labels(y_train)
    
    X_test = np.load(data_folder + 'X_test.npy')
    y_test = np.load(data_folder + 'Y_test.npy', allow_pickle=True)
    y_test = transform_labels(y_test)
    
    if verbose:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test
