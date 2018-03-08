import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def fit(train_x, train_y, test_x, test_y):
    # placeholders for inputs and outputs
    X = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, 3])

    # weight and bias
    weight = tf.Variable(tf.zeros([4, 3]))
    bias = tf.Variable(tf.zeros([3]))

    # output after going activation function
    output = tf.nn.softmax(tf.matmul(X, weight) + bias)
    # cost function
    cost = tf.reduce_mean(tf.square(Y - output))
    # train model
    train = tf.train.AdamOptimizer(0.01).minimize(cost)

    # check success and failures
    success = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(success, tf.float32)) * 100

    # initialize variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # start the tensorflow session
    with tf.Session() as sess:
        costs = []
        sess.run(init)
        # train model 1000 times
        for i in range(1000):
            _, c = sess.run([train, cost], {X: train_x, Y: [t for t in train_y.as_matrix()]})
            costs.append(c)

        print('Training finished!')

        # plot cost graph
        plt.plot(range(1000), costs)
        plt.title('Cost Variation')
        plt.savefig('graph.png')
        # plt.show()
        print('Accuracy: %.2f' % accuracy.eval({X: test_x, Y: [t for t in test_y.as_matrix()]}))

        # Save the variables to disk.
        save_path = saver.save(sess, './model.ckpt')


def predict(test_x, test_y):
    # reset all
    tf.reset_default_graph()

    # placeholders for inputs and outputs
    X = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, 3])

    # weight and bias
    weight = tf.Variable(tf.zeros([4, 3]))
    bias = tf.Variable(tf.zeros([3]))

    # output after going activation function
    output = tf.nn.softmax(tf.matmul(X, weight) + bias)

    # check success and failures
    success = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(success, tf.float32)) * 100

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # restore
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, './model.ckpt')
        print('Model restored.')

        # Check the values of the variables
        print('Accuracy: %.2f' % accuracy.eval({X: test_x, Y: [t for t in test_y.as_matrix()]}))


def main():
    # read data from csv
    train_data = pd.read_csv('iris_training.csv', names=['f1', 'f2', 'f3', 'f4', 'f5'])
    test_data = pd.read_csv('iris_test.csv', names=['f1', 'f2', 'f3', 'f4', 'f5'])

    # encode results to onehot
    train_data['f5'] = train_data['f5'].map({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]})
    test_data['f5'] = test_data['f5'].map({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]})

    # separate train data
    train_x = train_data[['f1', 'f2', 'f3', 'f4']]
    train_y = train_data.ix[:, 'f5']

    # separate test data
    test_x = test_data[['f1', 'f2', 'f3', 'f4']]
    test_y = test_data.ix[:, 'f5']

    fit(train_x, train_y, test_x, test_y)
    
    predict(test_x, test_y)


if __name__ == '__main__':
    # execute only if run as a script
    main()
