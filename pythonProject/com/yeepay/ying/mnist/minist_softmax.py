import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
  
x = tf.placeholder("float", shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
  
y = tf.nn.softmax(tf.matmul(x,W) + b)
 
y_ = tf.placeholder("float", shape=[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  
init = tf.initialize_all_variables()
sess.run(init)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)#梯度下降优化器
 
  
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
     
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

