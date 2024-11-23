import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adadelta, Adamax
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

#wandb.init(entity="safwennaimi", sync_tensorboard=True)





if __name__ == "__main__":
    

    crouch = np.load("E:/stm/Actions_keypoints/crouch.npy")
    look_tunnel = np.load("E:/stm/Actions_keypoints/look_tunnel.npy")
    sit_down = np.load("E:/stm/Actions_keypoints/sit_down.npy")
    standing = np.load("E:/stm/Actions_keypoints/standing.npy")
    UseCellPhone = np.load("E:/stm/Actions_keypoints/UseCellPhone.npy")
    walk = np.load("E:/stm/Actions_keypoints/walk.npy")


    crouch = np.array(crouch)
    look_tunnel = np.array(look_tunnel)
    sit_down = np.array(sit_down)
    standing = np.array(standing)
    UseCellPhone = np.array(UseCellPhone)
    walk = np.array(walk)


        # Reshape to (num_blocks, seq_len, num_keypoints, coords)
    num_blocks_1 = crouch.shape[0]
    seq_len_1 = crouch.shape[1]
    num_keypoints = 25 #25
    num_coords = 3
    crouch = crouch.reshape(num_blocks_1, seq_len_1, num_keypoints, num_coords)
    
    num_blocks_2 = look_tunnel.shape[0]
    seq_len_2 = look_tunnel.shape[1]
    num_keypoints = 25 
    num_coords = 3
    look_tunnel = look_tunnel.reshape(num_blocks_2, seq_len_2, num_keypoints, num_coords)
    
    # Reshape to (num_blocks, seq_len, num_keypoints, coords)
    num_blocks_3 = sit_down.shape[0]
    seq_len_3 = sit_down.shape[1]
    num_keypoints = 25 
    num_coords = 3
    sit_down = sit_down.reshape(num_blocks_3, seq_len_3, num_keypoints, num_coords)

    # Reshape to (num_blocks, seq_len, num_keypoints, coords)
    num_blocks_4 = standing.shape[0]
    seq_len_4 = standing.shape[1]
    num_keypoints = 25 
    num_coords = 3
    standing = standing.reshape(num_blocks_4, seq_len_4, num_keypoints, num_coords)

    # Reshape to (num_blocks, seq_len, num_keypoints, coords)
    num_blocks_5 = UseCellPhone.shape[0]
    seq_len_5 = UseCellPhone.shape[1]
    num_keypoints = 25  
    num_coords = 3
    UseCellPhone = UseCellPhone.reshape(num_blocks_5, seq_len_5, num_keypoints, num_coords)

    # Reshape to (num_blocks, seq_len, num_keypoints, coords)
    num_blocks_6 = walk.shape[0]
    seq_len_6 = walk.shape[1]
    num_keypoints = 25  #25
    num_coords = 3
    walk = walk.reshape(num_blocks_6, seq_len_6, num_keypoints, num_coords)

    




    Final = np.concatenate((crouch, look_tunnel, sit_down, standing, UseCellPhone, walk))
    print(Final.shape)
    



    crouch_label = np.zeros((698), dtype="int").reshape(-1,1)
    look_tunnel_label = np.ones((139), dtype="int").reshape(-1,1)
    sit_down_label = 2 * np.ones((1556), dtype="int").reshape(-1,1)
    standing_label = 3 * np.ones((3030), dtype="int").reshape(-1,1)
    UseCellPhone_label = 4 * np.ones((819), dtype="int").reshape(-1,1)
    walk_label = 5 * np.ones((2974), dtype="int").reshape(-1,1)
    



    labels = np.vstack([crouch_label, look_tunnel_label, sit_down_label, standing_label, UseCellPhone_label, walk_label])
    encoder = OneHotEncoder()
    y = encoder.fit_transform(labels).toarray().astype(int)
    #y = encoded_labels.toarray()
    num_classes = len(np.unique(y, axis=0))
    new_y = []
    for sample in y:
        label = np.where(sample==1)[0][0]
        new_y.append(label)
        
    y = np.array(new_y)

    
    X_train, X_test, y_train, y_test = train_test_split(Final, y, test_size = 0.20, shuffle=True)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    np.save('E:/stm/Actions_keypoints/X_train.npy', X_train)
    np.save('E:/stm/Actions_keypoints/X_test.npy', X_test) 
    np.save('E:/stm/Actions_keypoints/y_train.npy', y_train)
    np.save('E:/stm/Actions_keypoints/y_test.npy', y_test)
    
