## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# main.py
##################################################################################


import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from IE643_160050064_src import criterion, layers, model, optim, utils


PREFIX = 'IE643_160050064'


def run(config, X_train, labels_train):
    r"""
    Defines and trains a model according to the :attr:`config`
    on the data given by :attr:`X_train` and :attr:`labels_train`
    """
    if config['loss_fn'] == 'MSE':
        # One hot encode the labels
        y_train = np.zeros((labels_train.size, num_classes), dtype=np.long)
        y_train[np.arange(labels_train.size), labels_train] = 1
    elif config['loss_fn'] == 'CE':
        y_train = labels_train
    else:
        raise ValueError(f"Loss function of type '{config.loss_fn}' not implemented.")

    if config['val']:
        # Split data into training and validation sets
        val_size = int(config['val'] * X_train.shape[0])
        X_val, X_train = X_train[:val_size], X_train[val_size:]
        y_val, y_train = y_train[:val_size], y_train[val_size:]
        labels_val, labels_train = labels_train[:val_size], labels_train[val_size:]
        del val_size


    # Model definition
    net = model.Model()
    net.add_layer(layers.Linear(X_train.shape[1], 300))
    net.add_layer(layers.Sigmoid())
    net.add_layer(layers.Linear(300, 500))
    net.add_layer(layers.Sigmoid())
    net.add_layer(layers.Linear(500, 300))
    net.add_layer(layers.Sigmoid())
    net.add_layer(layers.Linear(300, num_classes))

    if config['loss_fn'] == 'MSE':
        net.add_layer(layers.Softmax())
        loss_fn = criterion.MSELoss()
    else:
        loss_fn = criterion.CrossEntropyLoss()

    dataloader_train = utils.DataLoader((X_train, y_train), batch_size=config['batch_size'], shuffle=True)
    optimizer = optim.SGD(net, config['lr'])


    # Train the model
    plot_x, plot_y = [], []
    for epoch in range(1, 1 + config['epochs']):
        net.train()
        loss = 0
        for X, y in dataloader_train:
            output = net(X)
            loss += loss_fn(output, y)
            net.zero_grad()
            grad = loss_fn.backward(output, y)
            net.backward(grad)
            optimizer.step()

        net.eval()
        if config['val'] is None:
            plot_x.append(epoch - 1)
            plot_y.append(loss / len(dataloader_train))
        elif epoch % 5 == 0:
            plot_x.append(epoch)
            output_train, output_val = net(X_train), net(X_val)
            if config['loss_fn'] == 'CE':
                output_train = layers.Softmax().forward(output_train)
                output_val = layers.Softmax().forward(output_val)
            train_error = 100 * (output_train.argmax(1) != labels_train).sum() / X_train.shape[0]
            val_error = 100 * (output_val.argmax(1) != labels_val).sum() / X_val.shape[0]
            plot_y.append((train_error, val_error))

    if config['val'] is None:
        net.eval()
        plot_x.append(epoch)
        plot_y.append(loss_fn(net(X_train), y_train))
        return plot_x[1:], plot_y[1:]
    else:
        plot_y = tuple(zip(*plot_y))
        return (plot_x, *plot_y)


def plot(config, param, values):
    r"""
    Plots the required graphs as per the question by varying
    :attr:`param` across the values present in :attr:`values`
    """
    plt.clf()
    ax = plt.subplot(111)
    cmap = plt.cm.cubehelix_r(np.linspace(0.2, 0.8, 2 * len(values)))
    for idx, value in enumerate(values):
        config[param] = value
        if config['val'] is None:
            plot_x, plot_y = run(config, X_train.copy(), labels.copy())
            plt.plot(plot_x, plot_y, color=cmap[2 * idx], label=f'{value}')
        else:
            plot_x, plot_y_train, plot_y_val = run(config, X_train.copy(), labels.copy())
            ax.plot(plot_x, plot_y_train, color=cmap[2 * idx], label=f'{value}_train')
            ax.plot(plot_x, plot_y_val, color=cmap[2 * idx + 1], label=f'{value}_val')
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss" if config['val'] is None else "Error Percentage (%)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


if __name__ == '__main__':

    # Parse the arguments
    parser = argparse.ArgumentParser(description="This script runs the experiments given in the assignment")
    parser.add_argument('--data', type=str, required=True, help="location of the training data")
    parser.add_argument('--experiments', nargs='+', help="IDs of experiments to run")
    args = parser.parse_args()


    # Read (possibly MNIST?) data from file
    data = np.genfromtxt(args.data, delimiter=',', skip_header=1)
    labels, X_train = data[:, :1], data[:, 1:]
    labels = labels.reshape(-1).astype(np.long)
    labels -= labels.min()
    num_classes = labels.max() + 1

    # Randomly shuffle the data
    indices = np.random.permutation(X_train.shape[0])
    X_train, labels = X_train[indices], labels[indices]
    del indices

    # Normalize the data
    X_train -= X_train.min(0, keepdims=True)
    X_train_max = X_train.max(0, keepdims=True)
    X_train = np.divide(X_train, X_train_max, out=np.zeros_like(X_train), where=(X_train_max != 0)) 
    del X_train_max


    # Run the experiments
    for exp_id in args.experiments:
        if exp_id == '1e':
            config = {'batch_size': 50, 'epochs': 100, 'lr': 1e-4, 'val': None}
            plot(config, 'loss_fn', ['MSE', 'CE'])
            plt.savefig(f'{PREFIX}_{exp_id}.png')

        if exp_id == '1f':
            config = {'batch_size': 50, 'epochs': 50, 'loss_fn': 'MSE', 'val': None}
            plot(config, 'lr', [1e-1, 1e-2, 1e-3, 1e-5, 1e-6])
            plt.savefig(f'{PREFIX}_{exp_id}.png')

        if exp_id == '1g':
            config = {'batch_size': 50, 'epochs': 50, 'loss_fn': 'CE', 'val': None}
            plot(config, 'lr', [1e-1, 1e-2, 1e-3, 1e-5, 1e-6])
            plt.savefig(f'{PREFIX}_{exp_id}.png')

        if exp_id == '1h':
            config = {'epochs': 50, 'val': None}
            for config['loss_fn'], config['lr'] in [('MSE', 1e-1), ('CE', 1e-2)]:
                plot(config, 'batch_size', [100, 200, 300, 400, 500])
                plt.savefig(f'{PREFIX}_{exp_id}_{config["loss_fn"]}.png')

        if exp_id == '2b':
            config = {'batch_size': 200, 'epochs': 50, 'loss_fn': 'MSE', 'val': 0.2}
            plot(config, 'lr', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
            plt.savefig(f'{PREFIX}_{exp_id}.png')

        if exp_id == '2c':
            config = {'batch_size': 200, 'epochs': 50, 'loss_fn': 'CE', 'val': 0.2}
            plot(config, 'lr', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
            plt.savefig(f'{PREFIX}_{exp_id}.png')
