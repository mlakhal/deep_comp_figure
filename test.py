import numpy as np

from utils.data import load_dataset

from models import CNN_1, CNN_2, CNN_3

def get_components(pr_result, db=0.5, comp=True):
    """ Return components that will be used 
    for our mesures(TP, FP/FN)

    Args:
        pr_result (numpy array)  : probability result matrix
        ds (float)               : decision boundary
        comp (boolean)           : working with (non)-compound

    Returns:
        int     : true positive.
        int     : false positive/negative

    """
    _, idxs = np.where(pr_result >= db)
    res = np.where(idxs == 0)[0] if comp else \
          np.where(idxs == 1)[0]
    tp = len(res)
    f_pos_or_neg = len(pr_result) - tp

    return tp, f_pos_or_neg

def main():
    path = "ds"
    btch_array = [4, 7]
    ds = load_dataset(path, btch_array)
    
    """
    model = CNN_1(nb_class=2, weights_path="CNN1_weights.h5")
    model = CNN_2(nb_class=2, weights_path="CNN2_weights.h5")
    """
    model = CNN_3(nb_class=2, weights_path="CNN3_weights.h5")
    proba = model.predict_proba(ds)
    n = len(btch_array) * 1000
    pr_comp = proba[:n]; pr_no_comp = proba[n:]

    tp, fn = get_components(pr_comp)
    mesure = lambda a,b: format(float(a) / (a + b), ".2f")
    _, fp = get_components(pr_no_comp, comp=False)
    precision = mesure(tp, fp)
    recall = mesure(tp, fn)
    print tp, fn, fp
    print '***'
    print precision
    print recall

if __name__ == "__main__":
    main()
