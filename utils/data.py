import numpy as np
import h5py

def shuffle_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    return a, b

def build_y(X):
    ns = len(X)
    middle = lambda x: len(x) / 2
    n = middle(X)
    
    Y = np.zeros((ns, 2))
    Y[:n, 0] = 1
    Y[n:, 1] = 1

    X, Y = shuffle_unison(X, Y)
    X_test, Y_test = shuffle_unison(X, Y)

    X = X.astype('float32')
    X /= 255
        
    return X, Y

def get_ds(X_train, X_test):
    X_train, Y_train = build_y(X_train)
    X_test, Y_test = build_y(X_test)

    return X_train, Y_train, X_test, Y_test

def load_dataset(path, nb):
    ''' Load the dataset.
    path  :: path to the dataset.
    nb    :: number of batches for each class.
    '''
    def load_batch(path, nb, idx):
        ''' Load batches for one class.
        idx :: class indice {0: 'COMP', 1: 'NOCOMP'}
        '''
        cls_name = 'COMP' if idx == 0 else 'NOCOMP'
        for i in nb:
            ds_name = path + '/ds_' + cls_name + '_' + str(i) + '.h5'
            with h5py.File(ds_name, 'r') as hf:
                data = hf.get('dataset_1')
                btch = np.array(data)
            ds = btch if i == nb[0] else np.vstack((ds, btch))

        return ds

    ds_comp = load_batch(path, nb, 0)
    ds_nocomp = load_batch(path, nb, 1)

    ds = np.vstack((ds_comp, ds_nocomp))
    
    return np.array(ds)
