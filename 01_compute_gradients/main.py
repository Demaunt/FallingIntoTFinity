import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[1,2], name="input")
y_ = tf.placeholder(tf.float32, shape=[1,1], name="target")

hw = tf.get_variable(name="hidden_weights", dtype=tf.float32, initializer=tf.constant([[0.1, 0.2]]))
hb = tf.get_variable(name="hidden_biases", dtype=tf.float32, initializer=tf.constant([0.3]))
ow = tf.get_variable(name="output_weights", dtype=tf.float32, initializer=tf.constant([[0.4]]))
ob = tf.get_variable(name="output_biases", dtype=tf.float32, initializer=tf.constant([0.5]))

'''
    скрытый слой
'''
hidden_potentials = tf.matmul(x_, hw, transpose_b=True) + hb
hidden_outputs = tf.sigmoid(hidden_potentials)

'''
    выходной слой
'''
output_potentials = tf.matmul(hidden_outputs, ow, transpose_b=True) + ob
output_outputs = tf.sigmoid(output_potentials)

'''
    ошибка
'''
error = y_ - output_outputs
mse = 0.5*error*error
cost = tf.reduce_mean(mse)

'''градиенты'''
'''grads = tf.gradients(ys=mse, xs=ow)'''
tf_grads = tf.train.GradientDescentOptimizer(learning_rate=0.1).compute_gradients(loss=cost, var_list=ow)

'''коррекции весов'''
delta_ow = tf.train.GradientDescentOptimizer(learning_rate=0.1).apply_gradients(grads_and_vars=tf_grads)

sess = tf.Session()
sess.run(tf.variables_initializer([hw, hb, ow, ob]))

print(sess.run(tf_grads, feed_dict={x_: [[1, 1]], y_: [[1]]}))
print(sess.run(ow))
sess.run(delta_ow, feed_dict={x_: [[1, 1]], y_: [[1]]})
print(sess.run(ow))
print()


