import numpy as np
import os
import datetime
from tensorflow import keras
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import tpe, hp, fmin
from hyperopt.pyll import scope

# Download the dataset (already implemented in keras-ncp)
(
    (x_train, y_train),
    (x_valid, y_valid),
) = kncp.datasets.icra2020_lidar_collision_avoidance.load_data()

def plot_lidar(lidar, ax):
    # Helper function for plotting polar-based lidar data
    angles = np.linspace(-2.35, 2.35, len(lidar))
    x = lidar * np.cos(angles)
    y = lidar * np.sin(angles)
    ax.plot(y, x)
    ax.scatter([0], [0], marker="^", color="black")
    ax.set_xlim((-6, 6))
    ax.set_ylim((-2, 6))

def train_ncp(params):

    print("x_train", str(x_train.shape))
    print("y_train", str(y_train.shape))
    
    # Plot the data
    sns.set()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    plot_lidar(x_train[0, 0, :, 0], ax1)
    plot_lidar(x_train[0, 12, :, 0], ax2)
    plot_lidar(x_train[9, 0, :, 0], ax3)
    ax1.set_title("Label: {:0.2f}".format(y_train[0, 0, 0]))
    ax2.set_title("Label: {:0.2f}".format(y_train[0, 12, 0]))
    ax3.set_title("Label: {:0.2f}".format(y_train[9, 0, 0]))
    fig.suptitle("LIDAR collision avoidance training examples")
    fig.savefig("lidar_examples.png")


    # Build the network
    N = x_train.shape[2]
    channels = x_train.shape[3]

    wiring = kncp.wirings.NCP(
        inter_neurons=params['nI'],  # Number of inter neurons
        command_neurons=params['nC'],  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=params['fS'],  # How many outgoing synapses has each sensory neuron
        inter_fanout=params['fI'],  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=params['rC'],  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=params['fM'],  # How many incomming syanpses has each motor neuron
    )
    rnn_cell = LTCCell(wiring)

    # We need to use the TimeDistributed layer to independently apply the
    # Conv1D/MaxPool1D/Dense over each time-step of the input time-series.
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(None, N, channels)),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(18, 5, strides=3, activation="relu")
            ),
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(20, 5, strides=2, activation="relu")
            ),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(keras.layers.Conv1D(22, 5, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(keras.layers.Conv1D(24, 5, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.Flatten()),
            keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
            keras.layers.RNN(rnn_cell, return_sequences=True),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss="mean_squared_error",
    )

    model.summary(line_length=100)

    '''# Plot the network architecture
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = rnn_cell.draw_graph(
        layout="spiral", neuron_colors={"command": "tab:cyan"}
    )
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("architecture.png")'''

    # Evaluate the model before training
    print("Validation set MSE before training")
    model.evaluate(x_valid, y_valid)

    #create log directory
    tag_name = str(params['nI']) +'-'+ str(params['nC']) +'-'+ str(params['fS']) +'-'+ str(params['fI']) +'-'+ str(params['rC']) +'-'+ str(params['fM'])
    log_path = os.path.join(os.getcwd(), "logs/fit/" + tag_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if len(os.listdir(log_path)) != 0:
        flag, i = True, 1
        while flag:
            if os.path.isdir(os.path.join(log_path,str(i))):
                i += 1
                continue
            else:
                os.makedirs(os.path.join(log_path,str(i)))
                flag = False
    log_dir = "logs/fit/" + tag_name +'/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard callback
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(
        x=x_train, y=y_train, batch_size=32, epochs=5, validation_data=(x_valid, y_valid), callbacks=[tensorboard_cb]
    )

    # Evaluate the model again after the training
    print("Validation set MSE after training")
    final_loss = model.evaluate(x_valid, y_valid)
    return final_loss

#HyperOpt
arch_space = {
    'nI': scope.int(hp.quniform('nI', 10, 16, q=1)),
    'nC': scope.int(hp.quniform('nC', 9, 15, q=1)),
    'fS': scope.int(hp.quniform('fS', 4, 10, q=1)), # must be at most the lower bound of nI
    'fI': scope.int(hp.quniform('fI', 3, 7, q=1)), # must be at most the lower bound of nC
    'fM': scope.int(hp.quniform('fM', 2, 8, q=1)), # must be at most the lower bound of nC
    'rC': scope.int(hp.quniform('rC', 1, 6, q=1)) # must be at most the lower bound of nC?
}
#print(type(3+2+1))
#print(type('nInter'+'nCommand'+1))
best_arch = fmin(fn=train_ncp, space=arch_space, algo=tpe.suggest, max_evals=100)
print(best_arch)