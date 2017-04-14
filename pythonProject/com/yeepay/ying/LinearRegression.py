import tensorflow as tf
from sklearn import datasets, linear_model

sess = tf.InteractiveSession()

x = []
y = []
def readfile(file):
    file = open(file,'r')
    for line in file:
        data = line.split(",")
        x.append([float(data[1])])
        y.append(float(data[2]))
    return x,y
x,y = readfile("data/input.txt")

# regr = linear_model.LinearRegression()
# regr.fit(x,y)
# print(regr.intercept_)
# print(regr.coef_)

train_x = tf.placeholder("float")
train_y = tf.placeholder("float")


w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
 
y_ = tf.add(tf.multiply(w,x),b)
loss = tf.reduce_mean(tf.square(y_-y))
  
init = tf.initialize_all_variables()
sess.run(init)
 
 
train = tf.train.GradientDescentOptimizer(0.000008).minimize(loss)
  
for i in range(10000):
    print(sess.run(w),sess.run(b))
    sess.run(train,{train_x:x,train_y:y})
    print(i,sess.run(loss,{train_x:x,train_y:y}),sess.run(w),sess.run(b))
