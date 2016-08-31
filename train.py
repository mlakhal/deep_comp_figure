"""

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py

================== ========== ============ =======  
     model         accuracy   precision    recall   
================== ========== ============ =======  
     CNN_1         0.7580     0.65         0.86
     CNN_2         0.7607     0.63         0.89 
     CNN_3         0.7588     0.64         0.88
                              
================== ========== ============ =======  

author: @mlakhal

"""

from utils.data import load_dataset, get_ds
from models import CNN_1, CNN_2, CNN_3

NB_EPOCH = 25
BATCH_SIZE = 100

def main():
    path = "ds"
    ds_tr = load_dataset(path, range(3))
    ds_ts = load_dataset(path, [5, 6])

    X_train, Y_train, X_test, Y_test = get_ds(ds_tr, ds_ts)
    
    """
    model = CNN_1(nb_class=2, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)
    model = CNN_2(nb_class=2, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)
    """
    model = CNN_3(nb_class=2, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)

    model.train(X_train, Y_train, X_test, Y_test)
    
    model.save_accuracy()
    model.save_loss()
    model.save_weights("CNN3_weights.h5")

if __name__ == "__main__":
    main()
