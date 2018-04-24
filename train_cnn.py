import time
import tensorflow as tf

########### Convolutional Neural Network Vlass ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # ======================================================================
    # Baseline model.
    # One fully connected layer.
    def model_1(self, X, hidden_size):

        X = tf.reshape(X, [-1, 28*28])

        return tf.contrib.layers.fully_connected(X,hidden_size,
            activation_fn = tf.nn.sigmoid)



    # ======================================================================
    # Model 2.
    # Use two convolutional layers.
    def model_2(self, X, hidden_size):

        conv1 = tf.layers.conv2d(inputs=X,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.sigmoid)
        print(conv1.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=1)
        print(pool1.shape)
        conv2 = tf.layers.conv2d(inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.sigmoid)
        print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=1)
        print(pool2.shape)
        pool2 = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

        return tf.contrib.layers.fully_connected(pool2,hidden_size,
            activation_fn = tf.nn.sigmoid)
    


    # ======================================================================
    # Model 3.
    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        conv1 = tf.layers.conv2d(inputs=X,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        print(conv1.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        print(pool1.shape)
        conv2 = tf.layers.conv2d(inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        print(pool2.shape)
        pool2 = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

        return tf.contrib.layers.fully_connected(pool2,hidden_size,
            activation_fn = tf.nn.relu)



    # ======================================================================
    # Model 4.
    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        conv1 = tf.layers.conv2d(inputs=X,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        print(conv1.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        print(pool1.shape)
        conv2 = tf.layers.conv2d(inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        print(pool2.shape)
        pool2 = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

        hidden1 = tf.contrib.layers.fully_connected(pool2,hidden_size,
            activation_fn = tf.nn.relu, 
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=.1))

        return tf.contrib.layers.fully_connected(hidden1,hidden_size,
            activation_fn = tf.nn.relu, 
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=.1))



    # ======================================================================
    # Model 5.
    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        f = open("output%d.txt" % self.mode,"w+")
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int32, [None])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X,hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)

            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            logits = tf.layers.dense(
                inputs = features, 
                units = 10,
                activation = tf.nn.softmax)

            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            onehot_labels = tf.one_hot(indices=tf.cast(Y, tf.int32), 
                depth=10)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
                logits=logits)
            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(onehot_labels,1))
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


            

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    string = (20 * '*') + 'epoch' + str(i + 1) + (20 * '*') + "\n"
                    print(string)
                    f.write(string)
                    start_time = time.time()
                    s = 0
                    printStep = train_size // 100
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        if (s % printStep) == 0:    
                            print("Finished data up to {} out of {}".format(e,train_size))
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))
                    f.write('the training took: %d(s)\n' % (end_time - start_time))
                    total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct / testX.shape[0]))
                    f.write('accuracy of the trained model %f\n' % (total_correct / testX.shape[0]))
                    print ()
                    f.write("\n")
                f.close()
                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) / testX.shape[0]





