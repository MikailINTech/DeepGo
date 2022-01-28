import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D,Reshape
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import golois
planes = 31
moves = 361
N = 10000
epochs = 50
batch = 64
filters = 32
expand = filters*4

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

def SE_Block(t,filters,ratio=4):
  se_shape= (1,1,filters)
  se = layers.GlobalAveragePooling2D()(t)
  se = layers.Reshape(se_shape)(se)
  se = layers.Dense(filters // ratio,use_bias=False)(se)
  se = layers.Activation('ReLU')(se)
  se = layers.Dense(filters, use_bias=False)(se)
  se = layers.Activation(tf.keras.activations.swish)(se)
  x = layers.multiply([t,se])

  return x


def bottleneck_block(x, kernel_size,expand=expand, squeeze=filters,activation=tf.keras.activations.swish,SE=True):
    m = layers.Conv2D(expand, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(x)
    m = layers.Activation(activation)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation(activation)(m)
    m = layers.DepthwiseConv2D(kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m2= layers.DepthwiseConv2D((1,1), padding='same',kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation(activation)(m)
    m2 = layers.BatchNormalization()(m2)
    if SE:
      m = SE_Block(m,expand)
    m = layers.Concatenate(axis=-1)([m, m2])
    m = layers.Conv2D(squeeze, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    if squeeze == x.shape[-1]:
      m = layers.Add()([m, x])
    return m

def buildModel(batch,epochs,filters,expand):
  input = keras.Input(shape=(19, 19, planes), name='board')
  x = layers.Conv2D(filters, (3, 3), padding='same',activation=tf.keras.activations.swish)(input)

  x = bottleneck_block(x,3,expand=filters,activation='ReLU')
  x = bottleneck_block(x,3,activation='ReLU',SE=False)
  x = bottleneck_block(x,3,expand=72,squeeze=24,activation='ReLU',SE=False)
  x = bottleneck_block(x,5,expand=82,squeeze=24,activation='ReLU',SE=False)
  x = bottleneck_block(x,5,expand=92,squeeze=40)
  x = bottleneck_block(x,5,expand=176,squeeze=40)
  x = bottleneck_block(x,5,expand=176,squeeze=40)
  x = bottleneck_block(x,5,expand=120,squeeze=48)
  x = bottleneck_block(x,5,expand=156,squeeze=48)
  x = bottleneck_block(x,5,expand=312,squeeze=96)
  x = bottleneck_block(x,5,expand=528,squeeze=96)
  x = bottleneck_block(x,5,expand=528,squeeze=96)

  policy_head = layers.Conv2D(1, 1, padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
  policy_head = layers.BatchNormalization()(policy_head)
  policy_head = layers.Activation(tf.keras.activations.swish)(policy_head)
  #policy_head = layers.AveragePooling2D()(policy_head)
  policy_head = layers.Conv2D(1, 1, padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(policy_head)
  policy_head = layers.Activation(tf.keras.activations.swish)(policy_head)
  policy_head = layers.Conv2D(1, 1, padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(policy_head)
  policy_head = layers.Flatten()(policy_head)
  policy_head = layers.Activation('softmax', name='policy')(policy_head)

  value_head = layers.GlobalAveragePooling2D()(x)
  value_head = layers.Dense(200, kernel_regularizer=regularizers.l2(0.0001),activation=tf.keras.activations.swish)(value_head)
  value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

  model = keras.Model(inputs=input, outputs=[policy_head, value_head])

  return model

model=buildModel(batch,epochs,filters,expand)

model.summary ()

optimizer=tf.keras.optimizers.Nadam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
)

model.compile(optimizer=optimizer,
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})


filepath = './Mikail_MobileNetv3.h5'

checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='policy_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

tf.test.gpu_device_name()

T_max=1001
eta_max=1e-3
eta_min=1e-7
for epoch in range (1, T_max):
    print ('epoch ' + str (epoch))
    golois.getBatch(input_data, policy, value, end, groups, epoch * N)
    lr = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    K.set_value(model.optimizer.learning_rate, lr)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch,callbacks=callbacks)
    if (epoch % 2 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print('------------------------------------------------------------------------------------------------------------')
        print (f"Loss {val[0]:.5f} | Policy loss {val[1]:.5f} | Value loss {val[2]:.5f} | Policy acc {val[3]:.5f} | Value MSE {val[4]:.5f} ")
        print('------------------------------------------------------------------------------------------------------------')
