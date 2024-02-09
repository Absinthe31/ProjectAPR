import tensorflow as tf
import librosa as lr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy.stats as sc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv1D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils import resample
from keras.models import Model
from keras.layers import Input
import logging
from logging import FileHandler
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def prob_to_class(array):
    return array.argmax()

#logger class that logs results into a json-like file
class JsonFormatter(logging.Formatter):

    @staticmethod
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def format(self, record) -> str:
        
        log_record = {
            'timestamp': self.formatTime(record),
            'model': record.getMessage()
        }
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        output = json.dumps(log_record) + ','

        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]
        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]

        output = output.replace("'",'"')
        
        return output

json_handler = FileHandler('logs_short.log')
json_handler.setLevel(logging.INFO)
json_formatter = JsonFormatter()
json_handler.setFormatter(json_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(json_handler)
logging.getLogger().setLevel(logging.INFO)

#logger function
def log_model(model=None, score=-1, acc=-1, acc_std=-1, f1=-1, f1_std=-1, model_name='', serialize=False, dataset=''):

    json_dict = { 'model_name' : model.__repr__() if model_name == '' else model_name }

    if score != -1:
        json_dict['score'] = score
    if acc != -1:
        json_dict['acc'] = acc
    if acc_std != -1:
        json_dict['acc_std'] = acc_std
    if f1 != -1:
        json_dict['f1'] = f1
    if f1_std != -1:
        json_dict['f1_std'] = f1_std
    if serialize == True:
        try:
            json_dict['obj'] = cloudpickle.dumps(model)   
        except:
            print("Error with object serialization")
    if dataset != '':
        json_dict['dataset'] = dataset

    return json_dict    

#import dataset
features_norm = pd.read_json('./features_norm_short.json')

#function that balances the dataset and creates train and test splits
def create_trainable_dataset(features_norm):

    X = features_norm.drop(['label'], axis=1)
    Y = features_norm.label
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
    
    X_combined = pd.concat([X_train,Y_train], axis = 1)
    
    X_0 = X_combined[X_combined.label == 0]
    X_1 = X_combined[X_combined.label == 1]
    X_2 = X_combined[X_combined.label == 2]
    X_3 = X_combined[X_combined.label == 3]
    X_4 = X_combined[X_combined.label == 4]
    X_5 = X_combined[X_combined.label == 5]
    X_6 = X_combined[X_combined.label == 6]
    X_7 = X_combined[X_combined.label == 7]
    
    n = X_combined.groupby(['label']).count().mfcc.sort_values(ascending=False)[0]
    
    X_0_s = resample(X_0, replace=True if len(X_0) < n else False, n_samples=n, random_state=42)
    X_1_s = resample(X_1, replace=True if len(X_1) < n else False, n_samples=n, random_state=42)
    X_2_s = resample(X_2, replace=True if len(X_2) < n else False, n_samples=n, random_state=42)
    X_3_s = resample(X_3, replace=True if len(X_3) < n else False, n_samples=n, random_state=42)
    X_4_s = resample(X_4, replace=True if len(X_4) < n else False, n_samples=n, random_state=42)
    X_5_s = resample(X_5, replace=True if len(X_5) < n else False, n_samples=n, random_state=42)
    X_6_s = resample(X_6, replace=True if len(X_6) < n else False, n_samples=n, random_state=42)
    X_7_s = resample(X_7, replace=True if len(X_7) < n else False, n_samples=n, random_state=42)
    
    final_X = pd.concat([X_0_s, X_1_s, X_2_s, X_3_s, X_4_s, X_5_s, X_6_s, X_7_s])
    
    final_X = final_X.reset_index().drop('index',axis=1).sample(frac=1)
    
    X_train = final_X.drop(['label'], axis=1)
    Y_train = final_X.label
    
    X_train_1 = X_train.mfcc
    X_train_2 = X_train.drop(['mfcc'],axis=1).drop(['chromagram'],axis=1)
    X_train_3 = X_train.chromagram
    X_test_1 = X_test.mfcc
    X_test_2 = X_test.drop(['mfcc'],axis=1).drop(['chromagram'],axis=1)
    X_test_3 = X_test.chromagram
    
    X_train_1 = np.array([np.array(val) for val in X_train_1])
    X_train_2 = np.array([np.array(val) for val in X_train_2.to_numpy()]).astype('float64')
    X_train_3 = np.array([np.array(val) for val in X_train_3])
    X_test_1 = np.array([np.array(val) for val in X_test_1])
    X_test_2 = np.array([np.array(val) for val in X_test_2.to_numpy()]).astype('float64')
    X_test_3 = np.array([np.array(val) for val in X_test_3])
    
    Y_train = np.array([np.array(val) for val in Y_train])
    Y_test = np.array([np.array(val) for val in Y_test])

    return X_train_1, X_train_2, X_train_3, X_test_1, X_test_2, X_test_3, Y_train, Y_test


X_train_1, X_train_2, X_train_3, X_test_1, X_test_2, X_test_3, Y_train, Y_test = create_trainable_dataset(features_norm)

del features_norm

mfcc_input = Input(shape=(12, 862, 1))

chromagram_input = Input(shape=(12, 862, 1))

other_features_input = Input(shape=(16,))    

#params for the CNN
params_dict = {
    'filters' : 20,
    'max_pooling' : (1,2),
    'dropout' : 0.2,
    'conv_1' : (7,400),
    'conv_2' : (4,200),
    'initial_learning_rate' : 0.0005,
    'decay_steps' : 100000,
    'batch_size' : 64,
    'epochs' : 200
}


conv1 = Conv2D(filters=params_dict['filters'], kernel_size=params_dict['conv_1'], 
               input_shape=(12, 862, 1), activation='relu')(mfcc_input)
conv1 = MaxPooling2D(pool_size=params_dict['max_pooling'])(conv1)
drop1 = Dropout(params_dict['dropout'])(conv1)
conv2 = Conv2D(filters=params_dict['filters'], kernel_size=params_dict['conv_2'], activation='relu')(drop1)
conv2 = MaxPooling2D(pool_size=params_dict['max_pooling'])(conv2)
drop2 = Dropout(params_dict['dropout'])(conv2)


first_part_output = Flatten()(drop2)

conv6 = Conv2D(filters=params_dict['filters'], kernel_size=params_dict['conv_1'], 
               input_shape=(12, 862, 1), activation='relu')(chromagram_input)
conv6 = MaxPooling2D(pool_size=params_dict['max_pooling'])(conv6)
drop6 = Dropout(params_dict['dropout'])(conv6)
conv7 = Conv2D(filters=params_dict['filters'], kernel_size=params_dict['conv_2'], activation='relu')(drop6)
conv7 = MaxPooling2D(pool_size=params_dict['max_pooling'])(conv7)
drop7 = Dropout(params_dict['dropout'])(conv7)

second_part_output = Flatten()(drop7)

merged_model = tf.keras.layers.concatenate([first_part_output, second_part_output, other_features_input])

dense1 = Dense((first_part_output.shape[1]+second_part_output.shape[1]+8)/2, activation ='relu')(merged_model)
predictions = Dense(8,  activation ='softmax')(dense1)

model = Model(inputs=[mfcc_input, chromagram_input, other_features_input], outputs=predictions)
print(model.summary())

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params_dict['initial_learning_rate'],
    decay_steps=params_dict['decay_steps'],
    decay_rate=0.96,
    staircase=True)


#checkpoint to keep best epoch only
checkpoint_filepath = './keras_checkpoints_short/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True,
                                    save_weights_only=True)

#compile the model
model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              metrics =['accuracy'])

#fit the model
model.fit([X_train_1, X_train_3, X_train_2], to_categorical(Y_train),
          batch_size = params_dict['batch_size'],
          epochs = params_dict['epochs'],
          verbose = 1,
          validation_data =([X_test_1, X_test_3, X_test_2], to_categorical(Y_test)),
          callbacks =[model_checkpoint_callback])

#load best epoch model
model.load_weights(checkpoint_filepath)

predictions = list(map(prob_to_class, model.predict([X_test_1, X_test_3, X_test_2])))
print(classification_report(Y_test,predictions, digits=3))

#logging function to file (logs_short.log)
logger.info(log_model(model_name=str(params_dict).replace("'",''), 
                      acc=classification_report(Y_test,predictions, digits=3, output_dict=True)['accuracy'], 
                      f1=classification_report(Y_test,predictions, digits=3, output_dict=True)['macro avg']['f1-score'], 
                      dataset='HAR-sounds', ))


#confusion matrix printing
import seaborn as sn
cf_matrix = confusion_matrix(Y_test, predictions)
labels=['taking_a_bath','brushing_teeth','making_coffee','cooking',
        'using_microwave_oven','washing_dishes','washing_hands','no_activity']
df_cm = pd.DataFrame(cf_matrix, index=labels,
                  columns=labels)
plt.figure(figsize = (5,3.5))
s = sn.heatmap(df_cm, annot=True)
s.set(xlabel='X-Axis', ylabel='Y-Axis')
plt.show()

