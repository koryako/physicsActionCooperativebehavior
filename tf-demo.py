import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+10

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise
with tf.name_scope('input'):
   x_datas=tf.placeholder(tf.float32,[None,1],name="x_input")
   y_datas=tf.placeholder(tf.float32,[None,1],name="y_input")
"""
x_datas=tf.placeholder(tf.float32)
y_datas=tf.placeholder(tf.float32)
"""
m=tf.constant([[3,3]])
m1=tf.constant([[2],[2]])
def add_layer(inputs,in_size,out_size,activation_function=None):
   with tf.name_scope('layer'):
       w=tf.Variable(tf.random_normal([in_size,out_size]))
       
       b=tf.Variable(tf.zeros([1,out_size])+0.1)
       y=tf.matmul(inputs,w)+b
       if activation_function is None:
          outputs=y
       else:
          outputs=activation_function(y)
       return outputs

l1=add_layer(x_datas,1,10,activation_function=tf.nn.relu)

y=add_layer(l1,10,1,activation_function=None)

product=tf.matmul(m,m1)
"""
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()

"""
state=tf.Variable(0,name='count')
one=tf.constant(1)

new=tf.add(state,one)
update=tf.assign(state,new)

init=tf.global_variables_initializer()


with tf.Session() as sess:
   sess.run(init)
   for _ in range(3):
       sess.run(update)
       print(sess.run(state))
   result= sess.run(product)
   print (result)
"""
w=tf.Variable(tf.random_uniform([1],-1.0,1.0))

b=tf.Variable(tf.zeros([1]),name="weight")
print (b.name)
y=w*x_datas+b

loss=tf.reduce_mean(tf.square(y-y_datas))
"""
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_datas),reduction_indices=[1]))


opt=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
writer=tf.summary.FileWriter('logs/',sess.graph)
#tensorboard --logdir='logs/'
sess.run(init)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#plt.savefig('1232.jpg')
#plt.ion()
#plt.show()
for step in range(1000):
   sess.run(opt,feed_dict={x_datas:x_data,y_datas:y_data})
   if step%20==0:
      print(sess.run(loss,feed_dict={x_datas:x_data,y_datas:y_data}))
      try:
          ax.lines.remove(lines[0])
      except Exception:
          pass

      
      prediction_value=sess.run(y,feed_dict={x_datas:x_data})
      lines=ax.plot(x_data,prediction_value,'r-',lw=5)
      #ax.lines.remove(line[0])
      #plt.pause(0.1)
      
    # print(step,sess.run(w),sess.run(b))
plt.savefig('2323.jpg')
