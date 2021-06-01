#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
import IPython
import IPython.display
from sklearn.metrics import mean_squared_error

import pandas as pd
pd.set_option('display.max_columns', None)

#tf.compat.v1.disable_eager_execution()

import tensorflow.keras.backend as tfk
from sklearn.model_selection import KFold
import random

import os


# In[17]:


# Read data
df = pd.read_csv('Simulated_Cashflow_Lit.txt',sep=';')
df.describe()


# In[18]:


# helpful stuff
column_indices = {c: i for i, c in enumerate(df.columns)}
pay00i = column_indices['Pay00']
open00i = column_indices['Open00']
payments = ['Pay00','Pay01','Pay02','Pay03','Pay04','Pay05','Pay06','Pay07','Pay08','Pay09','Pay10','Pay11']


# In[47]:


# GLOBAL PARAMETERS

MASK_VALUE = -9999

# feature selection
EMBED_FEATURES  = ['LoB','cc','inj_part']
STATIC_FEATURES = ['AY','AQ','age','RepDel']

EMBED_OUTPUT = [51,46,4]

n_embed = len(EMBED_FEATURES)
n_static = len(STATIC_FEATURES)

SIMPLE = False
BATCH_SIZE = 8
LEARN_RATE = 5e-4
EMBED_FORMAT = 'full'

EXPORT_NAME = 'embed'+EMBED_FORMAT+'_batch'+str(BATCH_SIZE)+'_LR'+str(LEARN_RATE)
if SIMPLE == True:
    EXPORT_NAME = EXPORT_NAME + "_simple"
    
EXPORT_NAME = EXPORT_NAME + "_masked"


# In[48]:


print(EXPORT_NAME)


# In[49]:


# Centering and scaling
mean = 0
#std = df.UC.std()
std = 1

df_scaled = df

for pay in payments:
    df_scaled[pay] = (df_scaled[pay]-mean)/std


# In[50]:


df_scaled.describe()


# In[51]:


# Test train val split
trainval_df = df_scaled.loc[df_scaled['k_year'] != df_scaled['RepDel']]
test_df = df_scaled.loc[df_scaled['k_year'] != 11]


# In[52]:


kf = KFold(n_splits=20, shuffle=True,random_state =0)
kf.get_n_splits(trainval_df)

train_indices = list()
val_indices = list()
for train_index, val_index in kf.split(trainval_df):
  train_indices.append(train_index)
  val_indices.append(val_index)

# currently training CV1
train_index = train_indices[0]
val_index = val_indices[0]

train_df = trainval_df.iloc[train_index]
val_df = trainval_df.iloc[val_index]


# In[53]:


# Divide test set
def preprocess_closed_claims(local_test_df):
    current_year = local_test_df['AY'].max()
    test_df_arr = np.array(local_test_df)
    closed_indices = []
    new_indices = []
    estimated_UC = []
    for i in range(len(local_test_df)):
        k = int(test_df_arr[i,column_indices['k_year']])
        if test_df_arr[i,open00i + k] == 0:
            closed_indices.append(i)
            cum_pay = np.sum(test_df_arr[i,pay00i:pay00i+k+1])
            estimated_UC.append(cum_pay*std)
        else:
            new_indices.append(i)
    
    closed_test_df = test_df.iloc[closed_indices,:].reset_index(drop = True)
    closed_test_df.loc[:,'Estimated_UC'] = estimated_UC
    new_test_df = test_df.iloc[new_indices,:].reset_index(drop = True)
    
    return closed_test_df, new_test_df


# In[54]:


closed_test_df,new_test_df = preprocess_closed_claims(test_df)


# In[55]:


difference = closed_test_df[(closed_test_df['UC'].round(2) != closed_test_df['Estimated_UC'].round(2))]
difference


# In[56]:


(difference.UC - difference.Estimated_UC).sum()


# In[57]:


closed_test_df.Estimated_UC.sum()


# In[58]:


# Functions
def make_trainval_dataset(df):
    
    # RESHAPING
    n = len(df)

    stat = np.array(df[STATIC_FEATURES])

    embed = np.array(df[EMBED_FEATURES])

    # starting index
    i_pay = pay00i
    i_open = i_pay + 12
    i_lit = i_open + 12

    df_arr = np.array(df)

    seq = np.zeros((n,11,3))

    response = df_arr[:,(i_pay+1):(i_pay+12)]

    KANCER = np.array(df['k_year'])

    for i in range(n):
      k = KANCER[i]
      seq[i,:,0] = df_arr[i,i_pay:(i_pay + 11)]
      seq[i,:,1] = df_arr[i,i_open:(i_open + 11)]
      seq[i,:,2] = df_arr[i,i_lit:(i_lit + 11)]
      if k != 11:
        seq[i,k:,:] = MASK_VALUE
        response[i,k:] = MASK_VALUE
        
        
    # MAKE DATASET
    inputs = list()
    # order of inputs: seq,stat, cc, inj,

    # sequence
    seq = tf.stack(seq)
    seq.set_shape([None,11,None])
    seq = tf.data.Dataset.from_tensor_slices(seq)
    inputs.append(seq)

    # static features
    stat = tf.stack(stat)
    stat = tf.data.Dataset.from_tensor_slices(stat)
    inputs.append(stat)
    
    # embedd features
    for i in range(n_embed):
      embed_in = tf.stack(embed[:,i])
      embed_in = tf.data.Dataset.from_tensor_slices(embed_in)
      inputs.append(embed_in)

    inputs = tuple(inputs)
    X = tf.data.Dataset.zip(inputs)

    response = tf.stack(response)
    y = tf.data.Dataset.from_tensor_slices(response)

    Xy = tf.data.Dataset.zip((X,y)).batch(BATCH_SIZE).shuffle(n)

    return Xy
    
    
def test_for_prediction(df):
    n = len(df)

    stat = np.array(df[STATIC_FEATURES])
    embed = np.array(df[EMBED_FEATURES])

    # starting index
    i_pay = pay00i
    i_open = i_pay + 12
    i_lit = i_open + 12

    df_arr = np.array(df)

    response = df_arr[:,(i_pay+1):(i_pay+12)]


    seq = np.zeros((n,11,3))

    KANCER = np.array(df['k_year'])
    for i in range(n):
      k = KANCER[i]
      seq[i,:,0] = df_arr[i,i_pay:(i_pay + 11)]
      seq[i,:,1] = df_arr[i,i_open:(i_open + 11)]
      seq[i,:,2] = df_arr[i,i_lit:(i_lit + 11)]

      if k != 10:
        seq[i,k+1:,:] = MASK_VALUE
    
    inputs = list()
    
    inputs.append(seq)
    inputs.append(stat)
    for i in range(n_embed):
        inputs.append(embed[:,i])
    
    return inputs


# In[59]:


#tf.config.run_functions_eagerly(False)
def deeptriangle(simple = False):
  tfk.clear_session()
  timesteps = 11

  seq_input = tf.keras.Input(shape = (timesteps,3,), name = 'sequence_input')

  static_features_input = tf.keras.Input(shape=(n_static,), name = "static_input")
  static_features_repeated = tf.keras.layers.RepeatVector(timesteps)(static_features_input)

  LoB_input = tf.keras.Input(shape = (1,), name = "LoB")
  LoB_embedding = tf.keras.layers.Embedding(4,3)(LoB_input)
  LoB_embedding = tf.keras.layers.Flatten()(LoB_embedding)
  LoB_embedding = tf.keras.layers.RepeatVector(timesteps)(LoB_embedding)

  company_code_input = tf.keras.Input(shape = (1,), name = "company_input")
  company_code_embedding = tf.keras.layers.Embedding(51,50)(company_code_input)
  company_code_embedding = tf.keras.layers.Flatten()(company_code_embedding)
  company_code_embedding = tf.keras.layers.RepeatVector(timesteps)(company_code_embedding)

  injury_part_input = tf.keras.Input(shape = (1,), name = "injury_input")
  injury_part_embedding = tf.keras.layers.Embedding(46,45)(injury_part_input)
  injury_part_embedding = tf.keras.layers.Flatten()(injury_part_embedding)
  injury_part_embedding = tf.keras.layers.RepeatVector(timesteps)(injury_part_embedding)
  
  encoded = tf.keras.layers.Masking(mask_value = -9999)(seq_input)

  if simple == False:
    encoded = tf.keras.layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2)(encoded)
    encoded = tf.keras.layers.RepeatVector(timesteps)(encoded) # is actually named decoded originally!
    
  decoded = tf.keras.layers.LSTM(128, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2)(encoded)

  concat = tf.keras.layers.Concatenate()([decoded, LoB_embedding, company_code_embedding, injury_part_embedding,static_features_repeated])


  feature = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = 64, activation = "relu"))(concat)
  
  feature = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate = 0.2))(feature)
  output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = 1, activation = "relu"), name = "ultimate_claim")(feature)

  model = tf.keras.Model(inputs = [seq_input, static_features_input, LoB_input, company_code_input, injury_part_input],
                         outputs = output,
                         name = "DeepTriangle")
  return model
    

# masked funktion benyttet af Kuo i Deep Triangle
def masked_mse(missing_value = MASK_VALUE):
    
    def custom_mse(y_true, y_pred):
    # assume 1st dimension is the number of samples
        keep= tfk.cast(tfk.not_equal(y_true, missing_value), tfk.floatx())
        mse = tfk.mean(tfk.square((y_pred-y_true)*keep), axis=2)

        return mse

    return custom_mse


def customMSE(masking_value = MASK_VALUE, RMSE = False):

  def teehee(y_true,y_pred):
    y_true = tfk.cast(y_true, 'float32')
    y_pred = tfk.cast(y_pred, 'float32')

    keep= tfk.cast(tfk.not_equal(y_true, masking_value), tfk.floatx())

    y_true = y_true*keep
    y_pred = y_pred*keep

    y_true = tfk.sum(y_true,axis = 1)
    y_pred = tfk.sum(y_pred,axis = 1)

    mse = tfk.mean(tfk.square(y_true-y_pred))
    if RMSE == True:
      mse = tfk.sqrt(mse)
    return mse


  return teehee


# In[64]:


# CALLBACKS
checkpoint_path = "deep_checkpoints/" + EXPORT_NAME + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Earlystopping callback
es_callback = tf.keras.callbacks.EarlyStopping(min_delta = 0.0, patience = 30, mode = "min", restore_best_weights = True,monitor='val_loss')

# CVSlogger
log_path = "deep_logs/" + EXPORT_NAME + ".log"
csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=",", append=True)


# In[65]:


def create_model():
  #if SIMPLE == True:
    #model = deeptriangle_simple()
  #else:
    #model = deeptriangle()
  model = deeptriangle(simple = SIMPLE)
  
  model.compile(loss=masked_mse(),
                optimizer=tf.optimizers.Adam(learning_rate = LEARN_RATE),
                metrics=masked_mse())
  model.get_config()
  model.from_config(model.get_config(),custom_objects={'loss':masked_mse(),'custom_mse':masked_mse()})

  return model


# In[66]:


# 10 embed 
model = create_model()

# datasets
train_set = make_trainval_dataset(train_df)
val_set = make_trainval_dataset(val_df)

test_prediction_set = test_for_prediction(new_test_df)

MAX_EPOCHS = 10
history = model.fit(train_set,
                    epochs = MAX_EPOCHS,
                    callbacks = [es_callback,cp_callback,csv_logger],
                    validation_data=val_set,
                    verbose = 1)

model_path = "deep_models/" + EXPORT_NAME
model.save(model_path)

predicted = model.predict(test_prediction_set)
predict_path  = "deep_predictions/" + EXPORT_NAME
np.save(predict_path,predicted)


# In[ ]:


# Different ways of reloading


# In[13]:


predict_path  = "deep_predictions/" + EXPORT_NAME + ".npy"
predicted = np.load(predict_path)


# In[44]:


model_path = "deep_models/" + EXPORT_NAME
model = tf.keras.models.load_model(model_path,custom_objects={'loss': customMSE(),'teehee':customMSE(-9999,True)})
model.compile(loss=customMSE(masking_value = -9999),
                optimizer=tf.optimizers.Adam(learning_rate = 0.0005),
                metrics=customMSE(masking_value = -9999, RMSE = True))


# In[47]:


prediction_inputs = test_for_prediction(new_test_df)
predicted = model.predict(prediction_inputs)


# In[20]:


np.shape(predicted)


# **Export dataframe of RNN predictions**

# In[35]:


def predicted_to_dataframe(predicted,new_test_df,closed_test_df):
    n1 = np.shape(predicted)[0]
    n2 = np.shape(predicted)[1]
    predicted = np.reshape(predicted,(n1,n2))
    predicted = np.sum(predicted, axis = 1)
    
    pay00 = new_test_df['Pay00']
    
    pred_UC = pay00 + predicted
    
    df1 = new_test_df.loc[:,('AY','RepDel','UC')]
    df1['Estimated_UC'] = pred_UC*std
    
    df2 = closed_test_df.loc[:,('AY','RepDel','UC','Estimated_UC')]
    
    df = pd.concat([df1,df2])
    
    return df


# In[36]:


np.count_nonzero(predicted)


# In[37]:


pickle_path = 'deep_pickles/' + EXPORT_NAME

RNN_df = predicted_to_dataframe(predicted,new_test_df,closed_test_df)


RNN_df.to_pickle(pickle_path) 


# In[38]:


RMSE = np.sqrt(((RNN_df.UC - RNN_df.Estimated_UC)**2).mean())


# In[40]:


RMSE


# In[35]:


df[(df['RepDel']==1) & (df['AY'] == 2) & (df['Open10'] == 1)]


# In[31]:


RNN_df.head(50)


# **Epoch progression Visualization**

# In[ ]:


#hist = np.load('deephistories/hist_embed10_CV1.npy', allow_pickle=True).item()
hist = pd.read_csv('deep_logs/embedfull_batch8_LR5e-4.log',sep = ',',engine = 'python')
hist2 = pd.read_csv('deep_logs/embedfull_batch8_LR5e-4_simple.log',sep = ',',engine = 'python')

plt.figure()
plt.plot(hist['val_teehee'])
plt.plot(hist2['val_teehee'])
plt.title('Model loss')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train','simple'], loc = 'upper left')
plt.show()


# In[ ]:





# In[ ]:




