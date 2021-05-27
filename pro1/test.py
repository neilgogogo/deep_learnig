import numpy as np
import dlc_practical_prologue as prologue

#Sometimes download directly will have http403 error
from six.moves import urllib
# have to add a header to your urllib request (due to that site moving to Cloudflare protection)
from Nets import MLP_Net, CNN_Net, CNN_one_by_one_Net,  CNN_Net_weight_sharing, \
    CNN_Net_weight_sharing_auxiliary_loss

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
#***********************
if __name__ == "__main__":
    # getting the training data and testing data
    N_PAIRS = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N_PAIRS)

    ##############################################################################
    #############################MLP##############################################
    ##############################################################################

    # combine two images into one 1*392 tensor
    tran_train_input=train_input.view(-1,2*14*14)
    tran_test_input=test_input.view(-1,2*14*14)

    # run the model
    tran_train_input=train_input.view(-1,2*14*14)
    tran_test_input=test_input.view(-1,2*14*14)
    my_model_1 = MLP_Net()
    # train the model
    my_model_1.trainer(tran_train_input, train_target,tran_test_input, test_target)
    # calculate the training and testing accuracy.
    model1_train_accuracy= my_model_1.compute_error(tran_train_input, train_target)
    model1_test_accuracy = my_model_1.compute_error(tran_test_input, test_target)

    #####################################################################################
    #############################Simple CNN##############################################
    #####################################################################################

    my_model_2 = CNN_Net()
    # train the model
    my_model_2.trainer(train_input, train_target, test_input, test_target)
    # calculate the training and testing accuracy.
    model2_train_accuracy = my_model_2.compute_error(train_input, train_target)
    model2_test_accuracy = my_model_2.compute_error(test_input, test_target)

    #####################################################################################
    #############################CNN to compare on by one################################
    #####################################################################################

    # prepocess the training and testing data
    tran_train_input = train_input.view([2000, 1, 14, 14])
    tran_train_classes = train_classes.view([2000])
    tran_test_input = test_input.view([2000, 1, 14, 14])
    tran_test_classes = test_classes.view([2000])

    my_model_3 = CNN_one_by_one_Net()
    # train the model
    my_model_3.trainer(tran_train_input, tran_train_classes)
    # calculate the training and testing accuracy.
    model3_train_accuracy = my_model_3.compare_two_digit(train_input, train_target)
    model3_test_accuracy = my_model_3.compare_two_digit(test_input, test_target)

    #####################################################################################
    #############################weights_sharing_CNN#####################################
    #####################################################################################

    my_model_5 = CNN_Net_weight_sharing()
    # train the model
    my_model_5.trainer(train_input, train_target, test_input, test_target)
    # calculate the training and testing accuracy.
    model5_train_accuracy = my_model_5.compute_error(train_input, train_target)
    model5_test_accuracy = my_model_5.compute_error(test_input, test_target)

    #####################################################################################
    ########################weights_sharing_auxiliary_loss_CNN###########################
    #####################################################################################

    # calculate the standard deviation:

    my_model_6 = CNN_Net_weight_sharing_auxiliary_loss()

    # train the model
    my_model_6.trainer(train_input, train_target, train_classes, test_input, test_target, test_classes)
    # calculate the training and testing accuracy.
    model6_train_accuracy = my_model_6.compute_error(train_input, train_target)
    model6_test_accuracy = my_model_6.compute_error(test_input, test_target)

    # print out the deviation and mean value of the training and testing errors
    print('MLP: train error:',  model1_train_accuracy)
    print('MLP: test error:',   model1_test_accuracy)
    print("MLP: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_1.parameters())))
    print('*******************************')
    print('Simple CNN: train error:', model2_train_accuracy)
    print('Simple CNN: test error:', model2_test_accuracy)
    print("Simple CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_2.parameters())))
    print('*******************************')
    print('CNN to compare on by one: train error: ',  model3_train_accuracy)
    print('CNN to compare on by one: test error:', model3_test_accuracy)
    print("CNN to compare on by one: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_3.parameters())))
    print('*******************************')
    print('weights_sharing_CNN: train error: ', model5_train_accuracy)
    print('weights_sharing_CNN: test error:', model5_test_accuracy)
    print("weights_sharing_CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_5.parameters())))
    print('*******************************')
    print('weights_sharing_auxiliary_loss_CNN: train error:', model6_train_accuracy)
    print('weights_sharing_auxiliary_loss_CNN: test error:',  model6_test_accuracy)
    print("weights_sharing_auxiliary_loss_CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_6.parameters())))