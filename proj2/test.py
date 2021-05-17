import math
import time
import torch
import matplotlib.pyplot as plt
from Loss_function import MSELoss
from Optimizer import SGD
from Sequential import Sequential


def plot_result(input_data, pred, true_label, model_name):
    """
    plot the points and the predicted result
    param input_data: points with x and y
    param pred: predicted result
    param target: the true label
    return the image drawing the predicted result
    """
    # 0 is the error predicted, 1 is the correct predicted
    error_data = (pred - true_label).eq(0).float()
    # in pred, 0 is outside the circle, 1 is inside the circle.
    # after +1: 1 is outside the circle, 2 is inside the circle.
    # after mul: 1 is outside the circle, 2 is inside the circle, 0 is false predicted
    target = (pred + 1).mul(error_data)
    cdict = {1: 'green', 2: 'blue', 0: 'red'}
    label_dict = {1: 'Outside the circle', 2: 'Inside the circle', 0: 'Error'}
    fig, ax = plt.subplots()
    for g in torch.unique(target):
        ix = torch.where(target.view(target.shape[0]) == g)
        ax.scatter(input_data[:, 0][ix],
                   input_data[:, 1][ix],
                   c=cdict[g.item()],
                   label=label_dict[g.item()])

    ax.legend()
    plt.axis("equal")
    plt.savefig(model_name + 'randompoint_target.png')
    # plt.show()
    # plt.close(fig)

def generate_set(num):
    """
    generate training data and testing data
    :param num: the number of pairs of random points and labels that you want to generate.
    :return: random_points: the random point in [0,1]
             targets:  whether the point is in the circle
    :return eaxmple: num=1, tensor([[0.8247, 0.7978]]), tensor([[0.]])
    """
#   using pytorch tensor operation
#   return the generated random points and labels
    random_points = torch.rand((num, 2))
    # check if the point is in the circle centered at (0.5,0.5) of radius 1/sqrt(2*math.pi)
    targets=torch.Tensor([int(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)<1/(2*math.pi)) for x in random_points]).view(-1,1)
    return random_points, targets


def train_model(model, train_input, train_target, test_input, test_target, learning_rate_list, batch_size=25,
                n_epochs=700, loss=MSELoss()):
    """
    Train model
    :param model: Sequence instance
    :param train_input: tensor of size [size, 2] with coordinates x,y of the samples
    :param train_target: tensor of size [size,1] with labels of the samples
    :param test_input: tensor of size [size, 2] with coordinates x,y of the samples
    :param test_target: tensor of size [size,1] with labels of the samples
    :param batch_size: (int) size of batch to perform SGD
    :param n_epochs: (int) number of iterations along the whole train set
    :param loss: loss function instance
    :param learning_rate: (float) which is used to perform SGD
    :return: my_loss: (list) values of loss along the epochs
    """
    # train model
    # time record
    start_time = time.time()
    train_loss_history = []
    train_accuracy = []
    test_loss_history = []
    test_accuracy = []

    sample_size = train_input.size(0)
    # reduce the learning rate when training epoch is halfway
    # so that the training loss can continue to decrease
    for epoch in range(n_epochs):
        for x in learning_rate_list:
            if epoch == x[1]:
                learning_rate = x[0]
        sgd = SGD(model.layers, learning_rate)
        cumulative_loss = 0
        for n_start in range(0, sample_size, batch_size):
            # resetting the gradients
            model.zero_grad()
            output = model(train_input[n_start: n_start + batch_size])
            # accumulating the loss over the mini-batches
            loss_ = loss(output, train_target[n_start: n_start + batch_size]) * batch_size
            cumulative_loss += loss_
            # calculating the gradient of the loss wrt final outputs
            loss_grad = loss.backward(output, train_target[n_start: n_start + batch_size])
            # propagating it backward
            model.backward(loss_grad)
            # updating the parameters
            sgd.step()
            print(model.model_name, ' | Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                  % (epoch + 1, epoch, n_start,
                     len(train_input), loss_))

        # get the training loss and accuracy
        train_pred = model(train_input)
        train_loss = loss(train_pred, train_target)
        train_loss_history.append(train_loss)
        train_accuracy_result = compute_accuracy(train_target, train_pred)
        train_accuracy.append(train_accuracy_result)

        # get the testing loss and accuracy
        test_pred = model(test_input)
        test_loss = loss(test_pred, test_target)
        test_loss_history.append(test_loss)
        test_accuracy_result = compute_accuracy(test_target, test_pred)
        test_accuracy.append(test_accuracy_result)

        # Printing the results of the current iteration
        print(model.model_name,
              " :the average training loss at epoch {} is {}".format(epoch + 1, (cumulative_loss / sample_size)),
              end='\r')
    print("\r")
    print(model.model_name,'Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print(model.model_name,"The final train accuracy is", compute_accuracy(train_target, train_pred))
    print(model.model_name,"The final test accuracy is ", compute_accuracy(test_target, test_pred))
    # Plotting the train and test loss and accuracy figure

    # Setting-up the plot
    fig=plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(1, 2, 1)

    ax2 = plt.subplot(1, 2, 2)

    # Drawing and labeling the curves
    ax1.plot(train_loss_history, label="Training Loss")
    ax1.plot(test_loss_history, label="Test Loss")

    # Adding the title and axis labels
    ax1.set_title('Train VS Test Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    #     #Saving the plot
    #     ax1.figure.savefig(model.model_name+'loss.png')

    # Drawing and labeling the curves
    ax2.plot(train_accuracy, label="Train Accuracy")
    ax2.plot(test_accuracy, label="Test Accuracy")

    # Adding the title and axis labels
    ax2.set_title('Train VS Test Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # Saving the plot
    #     ax2.figure.savefig(model.model_name+'accuracy.png')
    plt.savefig(model.model_name + 'accuracy_loss.png')
    # plt.show()
    # plt.close(fig)


def compute_accuracy(true_target, predicted):
    return (true_target - (predicted > 0.5).float()).eq(0).float().mean().item()


def compute_accuracy(true_target, predicted):
    return (true_target - (predicted > 0.5).float()).eq(0).float().mean().item()



if __name__ == "__main__":


    sample_size = 1000
    batch_size=50
    n_epochs=250
    # when the traing begin, the learning rate is 0.1;
    # After 100 epoch, the learning rate becomes 0.03
    # Then after 175 epoch, the learning rate becomes 0.01
    learning_rate=[(0.1,0),(0.03,100),(0.01,175)]


    train_input, train_target = generate_set(sample_size)
    test_input, test_target = generate_set(sample_size)

    # train the model with ReLU as the activation function
    layers = [
        {'type': 'Linear', 'shape': (2, 25)},
        {'type': 'ReLU'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'ReLU'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'ReLU'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'ReLU'},
        {'type': 'Linear', 'shape': (25, 1)}
    ]

    model = Sequential(layers,'model1')

    train_model(model, train_input, train_target,test_input, test_target, learning_rate, batch_size, n_epochs, loss=MSELoss())
    # draw the predicting plot
    # set the threshold to 0.5
    test_pred_1 = (model(test_input) > 0.5).float()
    plot_result(test_input, test_pred_1, test_target, 'model1')

    # train the model with Tanh as the activation function
    layers_2 = [
        {'type': 'Linear', 'shape': (2, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 1)}
    ]

    model_2 = Sequential(layers_2,'model2')

    train_model(model_2, train_input, train_target, test_input, test_target, learning_rate, batch_size, n_epochs,
                loss=MSELoss())
    # draw the predicting plot
    # set the threshold to 0.5
    test_pred_2 = (model_2(test_input) > 0.5).float()
    plot_result(test_input, test_pred_2, test_target, 'model2')

    learning_rate_3 = [(0.1, 0)]
    # train the model with Sigmois as the activation function
    layers_3 = [
        {'type': 'Linear', 'shape': (2, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Tanh'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Sigmoid'},
        {'type': 'Linear', 'shape': (25, 25)},
        {'type': 'Sigmoid'},
        {'type': 'Linear', 'shape': (25, 1)}
    ]


    model_3 = Sequential(layers_3, 'model3')

    train_model(model_3, train_input, train_target, test_input, test_target, learning_rate_3, batch_size, n_epochs,
                loss=MSELoss())
    # draw the predicting plot
    # set the threshold to 0.5
    test_pred_3 = (model_3(test_input) > 0.5).float()
    plot_result(test_input, test_pred_3, test_target, 'model3')