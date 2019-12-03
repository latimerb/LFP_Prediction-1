import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.stats as s

df = pd.read_csv("./data/model_fr_lfp.csv")

window_size = 100
pred_size = 10

n_feats = 5
df_class = np.zeros((n_feats,df.shape[0]-window_size-pred_size,window_size+2))
for i in np.arange(0,df.shape[0]-window_size-pred_size):
    df_class[0,i,0:window_size] = df['rawLFP'].iloc[i:i+window_size] 
    df_class[1,i,0:window_size] = df['PNA1'].iloc[i:i+window_size]
    df_class[2,i,0:window_size] = df['PNC1'].iloc[i:i+window_size]
    df_class[3,i,0:window_size] = df['ITN1'].iloc[i:i+window_size]
    df_class[4,i,0:window_size] = df['filtLFP'].iloc[i:i+window_size]
    
    df_class[:,i,-2] = np.mean(df['hilbLFP'].iloc[i+window_size:i+window_size+pred_size])


qn1 = np.quantile(df_class[0,:,-2],0.25)
qn2 = np.quantile(df_class[0,:,-2],0.50)
qn3 = np.quantile(df_class[0,:,-2],0.75)

df_class[:,df_class[0,:,-2]<=qn1,-1] = 0
df_class[:,(df_class[0,:,-2]>qn1) & (df_class[0,:,-2]<=qn2),-1] = 1
df_class[:,(df_class[0,:,-2]>qn2) & (df_class[0,:,-2]<=qn3),-1] = 2
df_class[:,df_class[0,:,-2]>qn3,-1] = 3

########### MODEL ###############

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(X,y):
    # load all train
    trainX, testX, trainy, testy = train_test_split(X,y)
    print(trainX.shape, trainy.shape)
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    pdb.set_trace()
    trainX, trainy, testX, testy = load_dataset(df_class[:,:,0:window_size],df_class[0,:,-1])
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()



############# PLOTTING ################

plt.figure()
ex = df_class[:,100,0:window_size]
plt.imshow(s.zscore(ex,axis=1))


plt.figure()
plt.hist(df_class[0,:,-2],bins=100)
plt.plot([qn1,qn1],[0,1200],'g')
plt.text(qn1-0.3,100,'Q1',bbox=dict(facecolor='red', alpha=0.5))
plt.plot([qn2,qn2],[0,1200],'g')
plt.text(qn2-0.3,100,'Q2',bbox=dict(facecolor='red', alpha=0.5))
plt.plot([qn3,qn3],[0,1200],'g')
plt.text(qn3-0.3,100,'Q3',bbox=dict(facecolor='red', alpha=0.5))

plt.text(qn3+0.2,100,'Q4',bbox=dict(facecolor='red', alpha=0.5))
plt.show()

