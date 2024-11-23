# GENERAL LIBRARIES
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import math
import numpy as np
import joblib
from pathlib import Path
# MACHINE LEARNING LIBRARIES
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#tf.executing_eagerly()
# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES
from utils.transformer import PatchClassEmbedding, SwinTransformerLayer
from utils.data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger
from wandb.keras import WandbCallback

import wandb
wandb.login()
wandb.init(entity="safwennaimi", project="STM_sLSTM_swin")
config = wandb.config
config.lr = 1e-3 #1e-5
config.batch_size = 4 #4
config.epochs = 100
config.dropout = 0.1
config.recurrent_dropout = 0.1
config.filters_1 = 16
config.filters_2 = 32
config.lstm = 16
config.learning_rate=0.0009524

WandbCallback = WandbCallback(
    monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
    log_weights=(False), log_gradients=(False), save_model=(False),
    training_data=None, validation_data=None, labels=None, predictions=36,
    generator=None, input_type=None, output_type=None, log_evaluation=(False),
    validation_steps=None, class_colors=None, log_batch_frequency=None,
    log_best_prefix="best_", save_graph=(False), validation_indexes=None,
    validation_row_processor=None, prediction_row_processor=None,
    infer_missing_processors=(True), log_evaluation_frequency=0,
    compute_flops=(False)
)



class CausalConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            dilation_rate=dilation_rate,
            **kwargs
        )

    def call(self, inputs):
        padded_inputs = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        conv_output = self.conv(padded_inputs)
        return conv_output


class BlockDiagonal(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0

        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks

        self.blocks = [
            tf.keras.layers.Dense(block_out_features)
            for _ in range(num_blocks)
        ]

    def call(self, x):
        x_chunks = tf.split(x, self.num_blocks, axis=-1)
        outputs = [block(x_i) for block, x_i in zip(self.blocks, x_chunks)]
        return tf.concat(outputs, axis=-1)

class sLSTMBlock(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4/3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor

        assert hidden_size % num_heads == 0
        assert proj_factor > 0

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.causal_conv = CausalConv1D(filters=1, kernel_size=4)

        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = tfa.layers.GroupNormalization(groups=num_heads)

        self.up_proj_left = tf.keras.layers.Dense(int(hidden_size * proj_factor))
        self.up_proj_right = tf.keras.layers.Dense(int(hidden_size * proj_factor))
        self.down_proj = tf.keras.layers.Dense(input_size)

    def call(self, x, prev_state):
        assert x.shape[-1] == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state
        x_norm = self.layer_norm(x)
        x_conv = tf.nn.silu(tf.squeeze(self.causal_conv(tf.expand_dims(x_norm, 1)), 1))

        z = tf.math.tanh(self.Wz(x) + self.Rz(h_prev))
        o = tf.math.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = tf.math.maximum(f_tilde + m_prev, i_tilde)
        i = tf.math.exp(i_tilde - m_t)
        f = tf.math.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = tf.nn.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)

class sLSTM(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=4/3):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor

        self.layers = [sLSTMBlock(input_size, hidden_size, num_heads, proj_factor) for _ in range(num_layers)]

    @tf.function
    def call(self, x, initial_state=None):
        tf.assert_rank(x, 3, message="Input must have 3 dimensions")
        if self.batch_first:
            x = tf.transpose(x, [1, 0, 2])
        
        seq_len = tf.shape(x)[0]
        batch_size = tf.shape(x)[1]

        if initial_state is not None:
            initial_state = tf.stack(list(initial_state))
            tf.assert_rank(initial_state, 4, message="Initial state must have 4 dimensions")
            num_hidden = tf.shape(initial_state)[0]
            state_num_layers = tf.shape(initial_state)[1]
            state_batch_size = tf.shape(initial_state)[2]
            state_input_size = tf.shape(initial_state)[3]
            tf.assert_equal(num_hidden, 4)
            tf.assert_equal(state_num_layers, self.num_layers)
            tf.assert_equal(state_batch_size, batch_size)
            tf.assert_equal(state_input_size, self.input_size)
            state = tf.transpose(initial_state, [1, 0, 2, 3])
        else:
            state = tf.zeros([self.num_layers, 4, batch_size, self.hidden_size])

        def body(t, state, output_ta):
            x_t = x[t]
            for layer in self.layers:
                x_t, state_tuple = layer(x_t, tuple(tf.unstack(state[self.layers.index(layer)])))
                state = tf.tensor_scatter_nd_update(state, [[self.layers.index(layer)]], [tf.stack(state_tuple)])
            output_ta = output_ta.write(t, x_t)
            return t + 1, state, output_ta

        _, final_state, output_ta = tf.while_loop(
            lambda t, *_: t < seq_len,
            body,
            [0, state, tf.TensorArray(tf.float32, size=seq_len)]
        )

        output = output_ta.stack()
        if self.batch_first:
            output = tf.transpose(output, [1, 0, 2])
        final_state = tuple(tf.unstack(tf.transpose(final_state, [1, 0, 2, 3])))
        return output, final_state


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return loss 


# TRAINER CLASS 
class Trainer:
    def __init__(self, config, logger, split=1, fold=0):
        self.config = config
        self.logger = logger
        self.split = split
        self.fold = fold
        self.trial = None
        self.bin_path = self.config['MODEL_DIR']
        
        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.pos_emb = self.config['POS_EMB']

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.config[self.config['DATASET']]['FRAMES'] // self.config['SUBSAMPLE'], 
                                            self.config[self.config['DATASET']]['KEYPOINTS'] * self.config['CHANNELS']))
        print(inputs.shape)
        
        # Define sLSTM layer configuration
        """
        slstm_layer = sLSTM(input_size=self.d_model, 
                            hidden_size=64,  # Add this line
                            num_heads=1, 
                            num_layers=3)
        
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x, _ = slstm_layer(x)  # Pass through sLSTM layer
        """
        
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)  
        x = tf.keras.layers.Dense(self.d_model)(x)
        #x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = transformer(x)
        print(x)
        x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)
        x = tf.keras.layers.Dense(64)(x)
        outputs = tf.keras.layers.Dense(6)(x)
        return tf.keras.models.Model(inputs, outputs)

    def get_model(self):
        transformer = SwinTransformerLayer(dim=self.d_model, depth=self.n_layers, num_heads=self.n_heads, window_size=30)
        self.model = self.build_act(transformer)
        print(self.model.summary())
        
        self.train_steps = np.ceil(float(self.train_len)/self.config['BATCH_SIZE'])
        self.test_steps = np.ceil(float(self.test_len)/self.config['BATCH_SIZE'])
        
        if self.config['SCHEDULER']:
            lr = CustomSchedule(self.d_model, 
                                warmup_steps=self.train_steps*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
                                decay_step=self.train_steps*self.config['N_EPOCHS']*self.config['STEP_PERC'])
        else:
            lr = 3 * 10**self.config['LR_MULT']
            ##lr = 1e-4
        
        optimizer = tfa.optimizers.LazyAdam(learning_rate=lr)
        #optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])
        

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

        self.name_model_bin = f"{self.config['MODEL_NAME']}_{self.config['MODEL_SIZE']}_{self.split}_{self.fold}.h5"
        """
        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.bin_path + self.name_model_bin,
                                                               monitor="val_accuracy",
                                                               save_best_only=False,
                                                               save_weights_only=False)
        """                                                            
        return

    
    
    def get_data(self):
        if self.config['DATASET'] == 'kinetics':
            train_gen, val_gen, test_gen, self.train_len, self.test_len = load_kinetics(self.config, self.fold)
            
            self.ds_train = tf.data.Dataset.from_generator(train_gen, 
                                                           output_types=('float32', 'uint8'))
            self.ds_val = tf.data.Dataset.from_generator(val_gen, output_types=('float32', 'uint8'))
            self.ds_test = tf.data.Dataset.from_generator(test_gen, output_types=('float32', 'uint8'))
            
        else:
            X_train, y_train, X_test, y_test = load_mpose(self.config['DATASET'], self.split, 
                                                          legacy=self.config['LEGACY'], verbose=False)
            print("the shape")
            print(X_train.shape)
            self.train_len = len(y_train)
            self.test_len = len(y_test)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=self.config['VAL_SIZE'],
                                                              random_state=self.config['SEEDS'][self.fold],
                                                              stratify=y_train)
                
            self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            self.ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            self.ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            
        self.ds_train = self.ds_train.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(X_train.shape[0])
        self.ds_train = self.ds_train.batch(self.config['BATCH_SIZE'])
        self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        
        self.ds_val = self.ds_val.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_val = self.ds_val.cache()
        self.ds_val = self.ds_val.batch(self.config['BATCH_SIZE'])
        self.ds_val = self.ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        
        self.ds_test = self.ds_test.map(lambda x,y : one_hot(x,y,self.config[self.config['DATASET']]['CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.batch(self.config['BATCH_SIZE'])
        self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        
    def get_random_hp(self):
        # self.config['RN_STD'] = self.trial.suggest_discrete_uniform("RN_STD", 0.0, 0.03, 0.01)
        self.config['WEIGHT_DECAY'] = self.trial.suggest_discrete_uniform("WD", 1e-5, 1e-3, 1e-5)
        self.config['N_EPOCHS'] = int(self.trial.suggest_discrete_uniform("EPOCHS", 50, 100, 10))
        self.config['WARMUP_PERC'] = self.trial.suggest_discrete_uniform("WARMUP_PERC", 0.1, 0.4, 0.1)
        self.config['LR_MULT'] = self.trial.suggest_discrete_uniform("LR_MULT", -5, -4, 1)
        self.config['SUBSAMPLE'] = int(self.trial.suggest_discrete_uniform("SUBSAMPLE", 4, 8, 4))
        self.config['SCHEDULER'] = self.trial.suggest_categorical("SCHEDULER", [False, False])
        
        # self.logger.save_log('\nRN_STD: {:.2e}'.format(self.config['RN_STD']))
        self.logger.save_log('\nEPOCHS: {}'.format(self.config['N_EPOCHS']))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.config['WARMUP_PERC']))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}'.format(self.config['WEIGHT_DECAY']))
        self.logger.save_log('LR_MULT: {:.2e}'.format(self.config['LR_MULT']))
        self.logger.save_log('SUBSAMPLE: {}'.format(self.config['SUBSAMPLE']))
        self.logger.save_log('SCHEDULER: {}\n'.format(self.config['SCHEDULER']))
        
    def do_training(self):
        self.get_data()
        self.get_model()


        self.model.fit(self.ds_train,
                       epochs=self.config['N_EPOCHS'], initial_epoch=0,
                       validation_data=self.ds_val,
                       callbacks=[WandbCallback], 
                       verbose=self.config['VERBOSE'],
                       #steps_per_epoch=int(self.train_steps*0.9),
                       #validation_steps=self.train_steps//9
                      )
        
        #self.model.load_weights(self.bin_path+self.name_model_bin)            
        _, accuracy_test = self.model.evaluate(self.ds_test, steps=self.test_steps)
        
        if self.config['DATASET'] == 'kinetics':
            g = list(self.ds_test.take(self.test_steps).as_numpy_iterator())
            X = [e[0] for e in g]
            X = np.vstack(X)
            y = [e[1] for e in g]
            y = np.vstack(y)
        else:
            X, y = tuple(zip(*self.ds_test))
        
        y_true = np.argmax(tf.concat(y, axis=0), axis=1)
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(tf.math.argmax(tf.concat(y, axis=0), axis=1), y_pred)

        text = f"Accuracy Test: {accuracy_test} <> Balanced Accuracy: {balanced_accuracy}\n"
        self.logger.save_log(text)

        #self.plot_confusion_matrix(y_true, y_pred, ['crouch','sit_down','standing','UseCellPhone','walk'])
        
        return y_true, y_pred, accuracy_test, balanced_accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentages
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # Display values as percentages with two decimal places
        plt.title('Confusion Matrix (in %)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.show()

    def print_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print("True\Pred", end="\t")
        for name in class_names:
            print(name[:7], end="\t")  # Print first 7 characters of each class name
        print()
        
        for i, row in enumerate(cm):
            print(f"{class_names[i][:7]}", end="\t")
            for cell in row:
                print(f"{cell}", end="\t")
            print()
        print()  # Add an extra newline for readability

    def print_classification_report(self, y_true, y_pred, class_names):
        report = classification_report(y_true, y_pred, target_names=class_names, digits=2)
        print("Classification Report:")
        print(report)
        print()  # Add an extra newline for readability

    
    def objective(self, trial):
        self.trial = trial     
        self.get_random_hp()
        _, bal_acc = self.do_training()
        return bal_acc
        
    def do_benchmark(self):
        all_y_true = []
        all_y_pred = []
        
        acc_list = []
        bal_acc_list = []

        self.class_names = ['crouch', 'look_tunnel', 'sit_dwon', 'standing', 'UseCellPhone', 'walk']

        for fold in range(self.config['FOLDS']):
            self.logger.save_log(f"- Fold {fold+1}")
            self.fold = fold
            
            y_true, y_pred, acc, bal_acc = self.do_training()

            acc_list.append(acc)
            bal_acc_list.append(bal_acc)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # Print confusion matrix and classification report after each fold
            print(f"\n--- Results for Fold {fold+1} ---")
            self.print_confusion_matrix(y_true, y_pred, self.class_names)
            self.print_classification_report(y_true, y_pred, self.class_names)
            
        np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + 'accuracy.npy', acc_list)
        np.save(self.config['RESULTS_DIR'] + self.config['MODEL_NAME'] + '_' + self.config['DATASET'] + 'balanced_accuracy.npy', bal_acc_list)

        self.logger.save_log(f"Accuracy mean: {np.mean(acc_list)}")
        self.logger.save_log(f"Accuracy std: {np.std(acc_list)}")
        self.logger.save_log(f"Balanced Accuracy mean: {np.mean(bal_acc_list)}")
        self.logger.save_log(f"Balanced Accuracy std: {np.std(bal_acc_list)}")

        # Print final results
        print("\n--- Final Results ---")
        self.print_confusion_matrix(all_y_true, all_y_pred, self.class_names)
        self.print_classification_report(all_y_true, all_y_pred, self.class_names)
        
    def do_random_search(self):
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.config['MODEL_NAME']),
                                         direction="maximize", pruner=pruner)
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.config['N_TRIALS'])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")

        self.logger.save_log("Best trial:")

        self.logger.save_log(f"  Value: {self.study.best_trial.value}")

        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        joblib.dump(self.study,
          f"{self.config['RESULTS_DIR']}/{self.config['MODEL_NAME']}_{self.config['DATASET']}_random_search_{str(self.study.best_trial.value)}.pkl")
        
    def return_model(self):
        return self.model