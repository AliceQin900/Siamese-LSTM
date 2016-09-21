from model.lstm import *
from preprocess import *


if __name__ == "__main__":

    pre_training = False # Set to true to load weights
    Syn_aug = False # it False faster but does slightly worse on Test dataset


    sls = lstm("lstm", training=True)

    if pre_training == True:
        print "Pre-training"
        train = pickle.load(open("stsallrmf.p", "rb"))
        sls.train_lstm(train, 66)
        print "Pre-training done"

    # Train Step
    train = pickle.load(open("semtrain.p", 'rb'))
    test = pickle.load(open("semtest.p", 'rb'))

    if Syn_aug == True:
        train = expand(train)
        sls.train_lstm(train, 375, test)
    else:
        sls.train_lstm(train, 330, test)

    # Test Step
    test = pickle.load(open("semtest.p", 'rb'))
    print sls.chkterr2(test)
