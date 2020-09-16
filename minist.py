from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend
#from tensorflow.keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import datetime
#load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0#normalizing
'''
x_train=tf.cast(x_train, dtype=tf.float16)
x_test = tf.cast(x_test, dtype=tf.float16)
y_train= tf.cast(y_train, dtype=tf.float16)
y_test=tf.cast(y_test, dtype=tf.float16)
print(y_test.dtype)
print(x_test.dtype)
tf.keras.backend.set_floatx('float16')
'''
# Add a channels dimension
x_train = x_train[..., tf.newaxis]#32*32->32*32*1
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(44)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(41)

#
#print(x_test.type)
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.conv1 = Conv2D(32, 3, padding='same', activation='relu')

    self.flatten = Flatten()
    print(self.flatten.get_weights()    )
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')


  def call(self, x):
    print(x.shape,'input size')
    x = self.conv1(x)
   # x=Conv2dd(x)
    print(x.shape,'afte conv')
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
      predictions = model(images)
      t_loss = loss_object(labels, predictions)
      test_loss(t_loss)
      test_accuracy(labels, predictions)

EPOCHS = 55
starttime=datetime.datetime.now()
for epoch in range(EPOCHS):
  # before next epoch start，reset evaluation metrcs
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()


  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
  oneruntime=datetime.datetime.now()
  print((oneruntime-starttime).seconds,'time')
''''''
