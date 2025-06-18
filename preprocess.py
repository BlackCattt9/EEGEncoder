
# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle


def load_BCI2a_data(data_path, subject, training, all_trials = True):
    """ Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training,
    and 288 x 9 trials in session 2 for testing.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts
    """

    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48
    window_Length = 7*250

    # Define MI trial window
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1


    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return


def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', n_classes = 4, isStandard = True, isShuffle = True):
    
    # Load and split the dataset into training and testing 

    if (dataset == 'BCI2a'):
        path = path + 's{:}/'.format(subject+1)
        X_train, y_train = load_BCI2a_data(path, subject+1, True)
        X_test, y_test = load_BCI2a_data(path, subject+1, False)

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = np.eye(n_classes)[y_train]

    # Prepare testing data
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = np.eye(n_classes)[y_test]

    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def pkl_save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def data_save():
    dataset = 'BCI2a'
    data_path = 'E:/datasets/BCI Competition IV/BCI Competition IV-2a/BCI Competition IV 2a mat/'
    n_sub = 9
    n_classes = 4
    # classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    # n_channels = 22
    # in_samples = 1125

    isStandard = 'True'

    for sub in range(n_sub):

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, n_classes=n_classes, isStandard=isStandard)

        data_to_save = (X_train, X_test, y_train_onehot, y_test_onehot)
        pkl_save(data_to_save, f'data/data_all_{sub+1}.pkl')


if __name__ == "__main__":
    data_save()

