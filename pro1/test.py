import numpy as np
import dlc_practical_prologue as prologue

#Sometimes download directly will have http403 error
from six.moves import urllib
# have to add a header to your urllib request (due to that site moving to Cloudflare protection)
from Nets import MLP_Net, CNN_Net, CNN_one_by_one_Net, ResNet, CNN_Net_weight_sharing, \
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

    # calculate the standard deviation:
    train_errors_1=[]
    test_errors_1=[]
    # run the model for 10 times and get the mean and standard deviation.
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N_PAIRS)
        tran_train_input=train_input.view(-1,2*14*14)
        tran_test_input=test_input.view(-1,2*14*14)
        my_model_1 = MLP_Net()
        # train the model
        my_model_1.trainer(tran_train_input, train_target,tran_test_input, test_target)
        train_errors_1.append(my_model_1.compute_error(tran_train_input, train_target))
        test_errors_1.append(my_model_1.compute_error(tran_test_input, test_target))



    #####################################################################################
    #############################Simple CNN##############################################
    #####################################################################################

    # calculate the standard deviation:
    train_errors_2 = []
    test_errors_2 = []
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            N_PAIRS)
        my_model_2 = CNN_Net()
        # train the model
        my_model_2.trainer(train_input, train_target, test_input, test_target)
        train_errors_2.append(my_model_2.compute_error(train_input, train_target))
        test_errors_2.append(my_model_2.compute_error(test_input, test_target))


    #####################################################################################
    #############################CNN to compare on by one################################
    #####################################################################################

    # prepocess the training and testing data
    tran_train_input = train_input.view([2000, 1, 14, 14])
    tran_train_classes = train_classes.view([2000])
    tran_test_input = test_input.view([2000, 1, 14, 14])
    tran_test_classes = test_classes.view([2000])

    # calculate the standard deviation:
    train_errors_3 = []
    test_errors_3 = []
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            N_PAIRS)
        my_model_3 = CNN_one_by_one_Net()
        # train the model
        my_model_3.trainer(tran_train_input, tran_train_classes)
        train_errors_3.append(my_model_3.compare_two_digit(train_input, train_target))
        test_errors_3.append(my_model_3.compare_two_digit(test_input, test_target))


    #####################################################################################
    ######################################ResNet#########################################
    #####################################################################################

    # calculate the standard deviation:
    train_errors_4 = []
    test_errors_4 = []
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            N_PAIRS)
        my_model_4 = ResNet()
        # train the model
        my_model_4.trainer(train_input, train_target, test_input, test_target)
        train_errors_4.append(my_model_4.compute_error(train_input, train_target))
        test_errors_4.append(my_model_4.compute_error(test_input, test_target))



    #####################################################################################
    #############################weights_sharing_CNN#####################################
    #####################################################################################

    # calculate the standard deviation:
    train_errors_5 = []
    test_errors_5 = []
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            N_PAIRS)
        my_model_5 = CNN_Net_weight_sharing()
        # train the model
        my_model_5.trainer(train_input, train_target, test_input, test_target)
        train_errors_5.append(my_model_5.compute_error(train_input, train_target))
        test_errors_5.append(my_model_5.compute_error(test_input, test_target))

    #####################################################################################
    ########################weights_sharing_auxiliary_loss_CNN###########################
    #####################################################################################

    # calculate the standard deviation:
    train_errors_6 = []
    test_errors_6 = []
    for num in range(10):
        N_PAIRS = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
            N_PAIRS)
        my_model_6 = CNN_Net_weight_sharing_auxiliary_loss()

        # train the model
        my_model_6.trainer(train_input, train_target, train_classes, test_input, test_target, test_classes)
        train_errors_6.append(my_model_6.compute_error(train_input, train_target))
        test_errors_6.append(my_model_6.compute_error(test_input, test_target))

    # print out the deviation and mean value of the training and testing errors
    print('MLP: The standard deviation of train error:', np.std(train_errors_1))
    print('MLP: The standard deviation of test error:', np.std(test_errors_1))
    print('MLP: The mean of train error:', np.mean(train_errors_1))
    print('MLP: The mean of test error:', np.mean(test_errors_1))
    print("MLP: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_1.parameters())))
    print('*******************************')
    print('Simple CNN: The standard deviation of train error:', np.std(train_errors_2))
    print('Simple CNN: The standard deviation of test error:', np.std(test_errors_2))
    print('Simple CNN: The mean of train error:', np.mean(train_errors_2))
    print('Simple CNN: The mean of test error:', np.mean(test_errors_2))
    print("Simple CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_2.parameters())))
    print('*******************************')
    print('CNN to compare on by one: The standard deviation of train error:', np.std(train_errors_3))
    print('CNN to compare on by one: The standard deviation of test error:', np.std(test_errors_3))
    print('CNN to compare on by one: The mean of train error: ', np.mean(train_errors_3))
    print('CNN to compare on by one: The mean of test error:', np.mean(test_errors_3))
    print("CNN to compare on by one: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_3.parameters())))
    print('*******************************')
    print('ResNet: The standard deviation of train error:', np.std(train_errors_4))
    print('ResNet: The standard deviation of test error:', np.std(test_errors_4))
    print('ResNet: The mean of train error: ', np.mean(train_errors_4))
    print('ResNet: The mean of test error:', np.mean(test_errors_4))
    print("ResNet: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_4.parameters())))
    print('*******************************')
    print('weights_sharing_CNN: The standard deviation of train error:', np.std(train_errors_5))
    print('weights_sharing_CNN: The standard deviation of test error:', np.std(test_errors_5))
    print('weights_sharing_CNN: The mean of train error: ', np.mean(train_errors_5))
    print('weights_sharing_CNN: The mean of test error:', np.mean(test_errors_5))
    print("weights_sharing_CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_5.parameters())))
    print('*******************************')
    print('weights_sharing_auxiliary_loss_CNN: The standard deviation of train error:', np.std(train_errors_6))
    print('weights_sharing_auxiliary_loss_CNN: The standard deviation of test error:', np.std(test_errors_6))
    print('weights_sharing_auxiliary_loss_CNN: The mean of train error:', np.mean(train_errors_6))
    print('weights_sharing_auxiliary_loss_CNN: The mean of test error:', np.mean(test_errors_6))
    print("weights_sharing_auxiliary_loss_CNN: The total number of the parameters is: %d" % (sum(p.numel() for p in my_model_6.parameters())))