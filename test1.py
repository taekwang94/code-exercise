import tensorflow as tf
tf.set_random_seed(1774)  # for reproducibility

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W1 = tf.Variable(tf.random_normal([1]), name="weight1")
b1 = tf.Variable(tf.random_normal([1]), name="bias1")
layer1 = tf.sigmoid(X*W1+b1)

# W2, b2 추가하니 성능 떨어짐
W2 = tf.Variable(tf.random_normal([1]), name="weight1")
b2 = tf.Variable(tf.random_normal([1]), name="bias1")


hypothesis = layer1 * W2 + b2


cost = tf.reduce_mean(tf.square(hypothesis - y_train))


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())


    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W2, b2],feed_dict={X:x_train,Y:y_train})

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
