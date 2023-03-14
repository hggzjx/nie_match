import tensorflow as tf 
import keras 
import numpy as np 
import os
import sys
sys.path.append('../')
from mie_match.mie_match import mie_match_1,mie_match_2
from dataloador.load_data import load_char_data,load_word_embed,load_char_embed,load_all_data

np.random.seed(1)
tf.random.set_seed(1)

mie_params = {
    'num_classes':2,
    'max_features':1700,
    'embed_size':300,
    'filters':300,
    'kernel_size':3,
    'strides':1,
    'padding':'same',
    'conv_activation_func':'relu',
    'embedding_matrix':[],
    'w_initializer':'random_uniform',
    'b_initializer':'zeros',
    'dropout_rate':0.2,
    'lstm_units':64,
    'mlp_activation_func':'relu',
    'mlp_num_layers':1,
    'mlp_num_units':128,
    'mlp_num_fan_out':128,
    'input_shapes':[(64,),(64,)],
    'batch_size_fit':64,
    'batch_size_eval':128,
    'epoch':15,
    'file':'',
    'task':'Classification',
}


char_embedding_matrix = load_char_embed(mie_params['max_features'],mie_params['embed_size'])
mie_params['embedding_matrix'] = char_embedding_matrix
params = mie_params
backend = mie_match_1(params)
file = params['file']
a, b, y = load_char_data('datasets/{}/train.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x = [a,b]
y = tf.keras.utils.to_categorical(y,num_classes=params['num_classes'])
a_eval, b_eval, y_eval = load_char_data('datasets/{}/dev.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x_eval = [a_eval,b_eval]
y_eval = tf.keras.utils.to_categorical(y_eval,num_classes=params['num_classes'])
a_test, b_test, y_test = load_char_data('datasets/{}/test.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x_test = [a_test,b_test]
y_test = tf.keras.utils.to_categorical(y_test,num_classes=params['num_classes'])
model = backend.build()
model.compile(
      loss='categorical_crossentropy', 
      optimizer='adam', 
      metrics=['accuracy']
      )
print(model.summary())

earlystop = keras.callbacks.EarlyStopping(
      monitor='val_accuracy', 
      patience=4, 
      verbose=2, 
      mode='max'
      )
bast_model_filepath = './output/best_mie_model.h5' 
checkpoint = keras.callbacks.ModelCheckpoint(
      bast_model_filepath, 
      monitor='val_accuracy', 
      verbose=1, 
      save_best_only=True,
      mode='max'
      )
model.fit(
      x=x, 
      y=y, 
      batch_size=params['batch_size_fit'], 
      epochs=params['epoch'], 
      validation_data=(x_eval, y_eval), 
      shuffle=True, 
      callbacks=[earlystop,checkpoint]
      )  
loss, acc = model.evaluate(
    x=x_test, 
    y=y_test, 
    batch_size=params['batch_size_eval' ],
    verbose=1
    )
print("Test loss:",loss, "Test accuracy:",acc)
