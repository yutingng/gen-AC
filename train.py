import pickle

def load_data(path, num_train=None, num_test=None):
    '''
    Loads dataset from `path` split into Pytorch train and test of 
    given sizes. Train set is taken from the front while
    test set is taken from behind.

    :param path: path to .p file containing data.
    '''
    f = open(path, 'rb')
    all_data = pickle.load(f)['samples']

    ndata_all = all_data.size()[0]
    
    if (num_train is None) and (num_test is None):
        num_train = np.floor(all_data*2/3)
        num_test = np.floor(all_data/3)
    elif (num_train is None) and (num_test==0):
        num_train = ndata_all
        
    assert num_train+num_test <= ndata_all

    train_data = all_data[:num_train]
    test_data = all_data[(ndata_all-num_test):]

    return train_data, test_data

def load_data2(path_train, path_test):

    train_data, _ = load_data(path_train, num_test=0)
    test_data, _ = load_data(path_test, num_test=0)
    
    return train_data, test_data