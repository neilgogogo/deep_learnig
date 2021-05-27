import torch
from torch import nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt



class MLP_Net(nn.Module):
    #     initialize the model
    def __init__(self):
        super(MLP_Net, self).__init__()
        self.linear1 = nn.Linear(2 * 14 * 14, 200)
        self.linear2 = nn.Linear(200, 50)
        #         self.linear3 = nn.Linear (200, 50)
        #         self.linear4 = nn.Linear (100, 50)
        self.linear5 = nn.Linear(50, 20)
        self.linear_out = nn.Linear(20, 2)
        # training parameter
        self.batch_size = 20
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)

    def forward(self, x):
        x = F.relu(self.linear1(x.view(-1, 2 * 14 * 14)))
        x = F.relu(self.linear2(x))
        #         x = F.relu(self.linear3(x))
        #         x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.linear_out(x)
        return x

    # Training Function

    def trainer(self, train_input, train_target, test_input, test_target):
        """
        Train the model on a training set
        :param train_input: Training features
        :param train_target: Training labels
        """
        start_time = time.time()
        train_loss_history = []
        test_loss_history = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(self.num_epochs):
            # train mode
            self.train()
            for batch_idx in range(0, train_input.size(0), self.batch_size):
                output = self(train_input[batch_idx:batch_idx + self.batch_size])
                loss = self.criterion(output, train_target[batch_idx:batch_idx + self.batch_size])
                self.optimizer.zero_grad()  # set gradients to zero
                loss.backward()  # backpropagation
                self.optimizer.step()
                # print the loss in every 50 epoch
                if not batch_idx % 500:
                    print('Model 1: MLP | Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                          % (epoch + 1, self.num_epochs, batch_idx,
                             len(train_input), loss))
            # test mode
            self.eval()
            # get the training loss and accuracy
            train_predicted = self(train_input)
            train_loss = self.criterion(train_predicted, train_target).item()
            train_loss_history.append(train_loss)
            _, train_pred = torch.max(train_predicted, 1)  # return the index of the bigger result
            train_accuracy_result = self.compute_accuracy(train_target, train_pred)
            train_accuracy.append(train_accuracy_result)

            # get the testing loss and accuracy
            test_predicted = self(test_input)
            test_loss = self.criterion(test_predicted, test_target).item()
            test_loss_history.append(test_loss)
            _, test_pred = torch.max(test_predicted, 1)  # return the index of the bigger result
            test_accuracy_result = self.compute_accuracy(test_target, test_pred)
            test_accuracy.append(test_accuracy_result)

            #       print the time used in one epoch
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        #       print the total training time used
        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

        # Setting-up the plot
        plt.figure(figsize=(15, 8))

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

        # Drawing and labeling the curves
        ax2.plot(train_accuracy, label="Train Accuracy")
        ax2.plot(test_accuracy, label="Test Accuracy")

        # Adding the title and axis labels
        ax2.set_title('Train VS Test Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        plt.savefig('./figure/model1_loss_accuracy.png')


        # Test error

    def compute_error(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: error rate of the input data
        """

        # test mode
        self.eval()
        outputs = self(input_data)
        _, predicted = torch.max(outputs, 1)
        return 1 - self.compute_accuracy(target, predicted)

    def compute_accuracy(self, target, pred):
        """
        Compute the training and testing error
        :param target: target data (whether 1 or 0)
        :param pred: predicted data
        :return
        """
        return (target - pred).eq(0).float().mean().item()

    def save_model(self, model_name):
        """
        Save the model to this folder
        :param model_name: the model name, e.g. CNN_Net.pth
        """
        torch.save(self, './model/' + model_name)


# Simple CNN
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 2)
        # parameters
        self.batch_size = 50
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = self.fc1(x.view(-1, 256))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return x

        # Training Function

    def trainer(self, train_input, train_target, test_input, test_target):
        """
        Train the model on a training set, and plot the loss and accuracy function
        Print the used time.
        :param train_input: Training input data
        :param train_target: Training labels
        :param test_input: Testing input data
        :param test_target: Testing labels
        """
        start_time = time.time()
        train_loss_history = []
        test_loss_history = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(self.num_epochs):
            self.train()
            for batch_idx in range(0, train_input.size(0), self.batch_size):
                output = self(train_input[batch_idx:batch_idx + self.batch_size])
                loss = self.criterion(output, train_target[batch_idx:batch_idx + self.batch_size])
                self.optimizer.zero_grad()  # set the weight and bias gradients to zero
                loss.backward()  # backpropagation
                self.optimizer.step()
                #                 # print the loss in every 50 epoch
                if not batch_idx % 500:
                    print('Model 2: Simple CNN |Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                          % (epoch + 1, self.num_epochs, batch_idx,
                             len(train_input), loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
            # test mode
            self.eval()
            # get the training loss and accuracy
            train_predicted = self(train_input)
            train_loss = self.criterion(train_predicted, train_target).item()
            train_loss_history.append(train_loss)
            _, train_pred = torch.max(train_predicted, 1)  # return the index of the bigger result
            train_accuracy_result = self.compute_accuracy(train_target, train_pred)
            train_accuracy.append(train_accuracy_result)

            # get the testing loss and accuracy
            test_predicted = self(test_input)
            test_loss = self.criterion(test_predicted, test_target).item()
            test_loss_history.append(test_loss)
            _, test_pred = torch.max(test_predicted, 1)  # return the index of the bigger result
            test_accuracy_result = self.compute_accuracy(test_target, test_pred)
            test_accuracy.append(test_accuracy_result)

        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

        # Setting-up the plot
        plt.figure(figsize=(15, 8))

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
        plt.savefig('./figure/model2_loss_accuracy.png')

        # Test error

    def compute_error(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: error rate of the input data
        """

        # test mode
        self.eval()
        outputs = self(input_data)
        _, predicted = torch.max(outputs, 1)
        return 1 - self.compute_accuracy(target, predicted)

    def compute_accuracy(self, target, pred):
        """
        Compute the training and testing error
        :param target: target data (whether 1 or 0)
        :param pred: predicted data
        :return
        """
        return (target - pred).eq(0).float().mean().item()

    def save_model(self, model_name):
        """
        Save the model to a direction
        :param model_name: the model name, e.g. CNN_Net.pth
        """
        torch.save(self, './model/' + model_name)


# CNN to compare two digits
class CNN_one_by_one_Net(nn.Module):
    def __init__(self):
        super(CNN_one_by_one_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        # parameters
        self.batch_size = 50
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.fc1(x.view(-1, 256))
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return x

        # Training Function

    def trainer(self, train_input, train_target):
        """
        Train the model on a training set
        :param train_input: Training features
        :param train_target: Training labels
        """
        start_time = time.time()
        self.train()
        for epoch in range(self.num_epochs):
            for batch_idx in range(0, train_input.size(0), self.batch_size):
                output = self(train_input[batch_idx:batch_idx + self.batch_size])
                loss = self.criterion(output, train_target[batch_idx:batch_idx + self.batch_size])
                self.optimizer.zero_grad()  # set gradients to zero
                loss.backward()  # backpropagate
                self.optimizer.step()
                #                 Every 50 data, output loss once
                if not batch_idx % 500:
                    print('Model 3 CNN_one_by_one_Net | Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                          % (epoch + 1, self.num_epochs, batch_idx,
                             len(train_input), loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

        # Test error

    def compute_error(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: error rate of the input data
        """
        # test mode
        self.eval()
        outputs = self(input_data)
        _, predicted = torch.max(outputs, 1)
        return 1 - self.compute_accuracy(target, predicted)

    def compute_accuracy(self, target, pred):
        """
        Compute the training and testing error
        :param target: target data (whether 1 or 0)
        :param pred: predicted data
        :return
        """
        return (target - pred).eq(0).float().mean().item()

    def compare_two_digit(self, input_data, comp_targets):

        # test mode
        self.eval()
        errors = 0
        for pairs, comp_target in zip(input_data, comp_targets):
            input_num1 = pairs[0].view([1, 1, 14, 14])
            input_num2 = pairs[1].view([1, 1, 14, 14])
            output_1 = self(input_num1)
            output_2 = self(input_num2)
            _, predicted_1 = torch.max(output_1, 1)  # return value and key
            _, predicted_2 = torch.max(output_2, 1)  # return value and key
            if (predicted_2 - predicted_1 > 0):
                result = 1
            else:
                result = 0
            if (comp_target != result):
                errors = errors + 1
        return float(errors) / input_data.size(0)

    def save_model(self, model_name):
        """
        Save the model to a direction
        :param model_name: the model name, e.g. CNN_Net.pth
        """
        torch.save(self, './model/' + model_name)

# weights_sharing_CNN
class CNN_Net_weight_sharing(nn.Module):
    def __init__(self):
        super(CNN_Net_weight_sharing, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)
        # parameters
        self.batch_size = 50
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)

    def forward(self, x):

        img1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        img2 = x[:, 1, :, :].view(-1, 1, 14, 14)

        img1 = F.relu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2))
        img1 = F.relu(F.max_pool2d(self.bn2(self.conv2(img1)), kernel_size=2, stride=2))
        img2 = F.relu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2))
        img2 = F.relu(F.max_pool2d(self.bn2(self.conv2(img2)), kernel_size=2, stride=2))
        output = torch.cat((img1.view(-1, 256), img2.view(-1, 256)), 1)
        output = self.fc1(output)
        output = F.dropout(output, training=self.training)
        output = F.relu(output)
        output = self.fc2(output)

        return output

        # Training Function

    def trainer(self, train_input, train_target, test_input, test_target):
        """
        Train the model on a training set
        :param train_input: Training features
        :param train_target: Training labels
        """

        start_time = time.time()
        train_loss_history = []
        test_loss_history = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(self.num_epochs):
            self.train()
            for batch_idx in range(0, train_input.size(0), self.batch_size):
                output = self(train_input[batch_idx:batch_idx + self.batch_size])
                target = train_target[batch_idx:batch_idx + self.batch_size]
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()  # set the weight and bias gradients to zero
                loss.backward()  # backpropagation
                self.optimizer.step()
                # print the loss in every 50 epoch
                if not batch_idx % 500:
                    print('Model 5: CNN_Net_weight_sharing | Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                          % (epoch + 1, self.num_epochs, batch_idx,
                             len(train_input), loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

            # test mode
            self.eval()
            # get the training loss and accuracy
            train_predicted = self(train_input)
            train_loss = self.criterion(train_predicted, train_target).item()
            train_loss_history.append(train_loss)
            _, train_pred = torch.max(train_predicted, 1)  # return the index of the bigger result
            train_accuracy_result = self.compute_accuracy(train_target, train_pred)
            train_accuracy.append(train_accuracy_result)

            # get the testing loss and accuracy
            test_predicted = self(test_input)
            test_loss = self.criterion(test_predicted, test_target).item()
            test_loss_history.append(test_loss)
            _, test_pred = torch.max(test_predicted, 1)  # return the index of the bigger result
            test_accuracy_result = self.compute_accuracy(test_target, test_pred)
            test_accuracy.append(test_accuracy_result)

        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

        # Setting-up the plot
        plt.figure(figsize=(15, 8))

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
        plt.savefig('./figure/model5_loss_accuracy.png')

        # Test error

    def compute_error(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: error rate of the input data
        """
        # test mode
        self.eval()
        outputs = self(input_data)
        _, predicted = torch.max(outputs, 1)
        return 1 - self.compute_accuracy(target, predicted)

    def compute_accuracy(self, target, pred):
        """
        Compute the training and testing error
        :param target: target data (whether 1 or 0)
        :param pred: predicted data
        :return
        """
        return (target - pred).eq(0).float().mean().item()

    def save_model(self, model_name):
        """
        Save the model to a direction
        :param model_name: the model name, e.g. CNN_Net.pth
        """
        torch.save(self, './model/' + model_name)


# weights_sharing_CNN
# What the weight sharing does is using both the 2000 images together to train the same layer,
# which will be better than purely using 1000 images
class CNN_Net_weight_sharing_auxiliary_loss(nn.Module):
    def __init__(self):
        super(CNN_Net_weight_sharing_auxiliary_loss, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 2)
        # parameters
        self.batch_size = 50
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)

    def forward(self, x):
        img1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        img2 = x[:, 1, :, :].view(-1, 1, 14, 14)

        # weight sharing
        img1 = F.relu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2))
        img1 = F.relu(F.max_pool2d(self.bn2(self.conv2(img1)), kernel_size=2, stride=2))
        img2 = F.relu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2))
        img2 = F.relu(F.max_pool2d(self.bn2(self.conv2(img2)), kernel_size=2, stride=2))

        #       detect the img1 figure

        output1 = img1.view(-1, 256)
        output1 = self.fc1(output1)
        output1 = F.dropout(output1, p=0.5, training=self.training)
        output1 = F.relu(output1)
        output1 = self.fc2(output1)
        #       detect the img2 figure
        output2 = img2.view(-1, 256)
        output2 = self.fc1(output2)
        output2 = F.dropout(output2, p=0.5, training=self.training)
        output2 = F.relu(output2)
        output2 = self.fc2(output2)

        output = torch.cat((output1, output2), 1)
        output = F.relu(F.dropout(self.fc3(output), p=0.5, training=self.training))
        output = F.relu(F.dropout(self.fc4(output), p=0.5, training=self.training))
        output = self.fc5(output)

        return output, output1, output2

        # Training Function with auxiliary_loss

    def trainer(self, train_input, train_target, train_classes, test_input, test_target, test_classes):
        """
        Train the model on a training set
        :param train_input: Training features
        :param train_target: Training labels
        :param train_classes: Training classes
        :param test_input: Testing features
        :param test_target: Training labels
        :output the loss plot
        """
        start_time = time.time()
        #         self.train()
        train_loss_history = []
        test_loss_history = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(self.num_epochs):
            # train mode
            self.train()
            for batch_idx in range(0, train_input.size(0), self.batch_size):
                output, output1, output2 = self(train_input[batch_idx:batch_idx + self.batch_size])
                #                 output = self(train_input[batch_idx:batch_idx+self.batch_size])
                target = train_target[batch_idx:batch_idx + self.batch_size]
                #                 print(output.shape)
                #                 print(target.shape)
                class1 = train_classes[batch_idx:batch_idx + self.batch_size, 0]
                class2 = train_classes[batch_idx:batch_idx + self.batch_size, 1]
                loss = self.criterion(output, target) + 0.5 * self.criterion(output1, class1) + 0.5 * self.criterion(
                    output2, class2)
                # gradients to zero
                self.optimizer.zero_grad()
                # backpropagation
                loss.backward()
                self.optimizer.step()
                #                 every 50 batch_idx, output the loss
                if not batch_idx % 500:
                    print('Model 6: CNN_Net_weight_sharing_auxiliary_loss | Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.6f'
                          % (epoch + 1, self.num_epochs, batch_idx,
                             len(train_input), loss))
            # test mode
            self.eval()
            # get the training loss and accuracy
            train_predicted, train_output1, train_output2 = self(train_input)
            train_loss = self.criterion(train_predicted, train_target) + 0.5 * self.criterion(train_output1,
                                                                                              train_classes[:,
                                                                                              0]) + 0.5 * self.criterion(
                train_output2, train_classes[:, 1])
            train_loss_history.append(train_loss.item())
            _, train_pred = torch.max(train_predicted, 1)  # return the index of the bigger result
            train_accuracy_result = self.compute_accuracy(train_target, train_pred)
            train_accuracy.append(train_accuracy_result)

            # get the testing loss and accuracy
            test_predicted, test_output1, test_output2 = self(test_input)
            test_loss = self.criterion(test_predicted, test_target) + 0.5 * self.criterion(test_output1, test_classes[:,
                                                                                                         0]) + 0.5 * self.criterion(
                test_output2, test_classes[:, 1])
            test_loss_history.append(test_loss.item())
            _, test_pred = torch.max(test_predicted, 1)  # return the index of the bigger result
            test_accuracy_result = self.compute_accuracy(test_target, test_pred)
            test_accuracy.append(test_accuracy_result)

            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

        # plot the accuracy and loss figure
        # Plotting the train and test loss and accuracy figure

        # Setting-up the plot
        plt.figure(figsize=(15, 8))

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
        #save the image
        plt.savefig('./figure/model6_loss_accuracy.png')

        # Test error

    def compute_error(self, input_data, target):
        """
        Compute the number of error of the model on a test set with batch_size
        :param input_data: test features
        :param target: test target
        :return: error rate of the input data
        """

        # test mode
        self.eval()
        outputs, _, _ = self(input_data)
        _, predicted = torch.max(outputs, 1)
        return 1 - self.compute_accuracy(target, predicted)

    def compute_accuracy(self, target, pred):
        """
        Compute the training and testing error
        :param target: target data (whether 1 or 0)
        :param pred: predicted data
        :return
        """
        return (target - pred).eq(0).float().mean().item()

    def save_model(self, model_name):
        """
        Save the model to a direction
        :param model_name: the model name, e.g. CNN_Net.pth
        :output the model pth.
        """
        torch.save(self, './model/' + model_name)
