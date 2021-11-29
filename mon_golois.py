import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import golois

planes = 31
moves = 361
N = 10000
epochs = 5
batch = 128
filters = 32
expand = 128


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

def bottleneck_block(x, expand=expand, squeeze=filters):
    m = layers.Conv2D(expand, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)
    m1 = layers.DepthwiseConv2D((3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m2= layers.DepthwiseConv2D((1,1), padding='same',kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m3= layers.DepthwiseConv2D((5,5),padding='same',kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m4= layers.DepthwiseConv2D((7,7),padding='same',kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m1 = layers.BatchNormalization()(m1)
    m1 = layers.Activation('relu')(m1)
    m2 = layers.BatchNormalization()(m2)
    m2 = layers.Activation('relu')(m2)
    m3 = layers.BatchNormalization()(m3)
    m3 = layers.Activation('relu')(m3)
    m4 = layers.BatchNormalization()(m4)
    m4 = layers.Activation('relu')(m4)
    m = layers.Concatenate(axis=-1)([m1, m2,m3,m4])
    m = layers.Conv2D(squeeze, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Add()([m, x])
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)
   
    return m

def buildModel(batch,epochs,filters,expand):
  input = keras.Input(shape=(19, 19, planes), name='board')
  x = layers.Conv2D(filters, 1, activation='relu', padding='same')(input)
  for i in range (30):
      x = bottleneck_block(x,expand,filters)
  policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
  policy_head = layers.Flatten()(policy_head)
  policy_head = layers.Activation('softmax', name='policy')(policy_head)
  value_head = layers.GlobalAveragePooling2D()(x)
  value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
  value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

  model = keras.Model(inputs=input, outputs=[policy_head, value_head])

  return model

model=buildModel(batch,epochs,filters,expand)

model.summary ()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
              


filepath = 'Mkt_model.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.00001)
callbacks = [checkpoint,reduce_lr]

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch)
    if (i % 2 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)

#model.save ('test.h5')
