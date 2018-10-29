import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

import pickle
import time

################ PARAMETERS #################
train_positive_lst_file = "electron.cv.txt"
train_positive_dir = "pssm/electron/"
train_negative_lst_file = "transport.cv.txt"
train_negative_dir = "pssm/transport/"

val_positive_lst_file = "electron.ind.txt"
val_positive_dir = "pssm/electron/"
val_negative_lst_file = "transport.ind.txt"
val_negative_dir = "pssm/transport/"


file_model = "model/elec_ind_conv200.model"
######## GLOBAL HYPER PARAMETERS ###############
CONV1D_FEATURE_SIZE = 200
CONV1D_KERNEL_SIZE = 3
AVGPOOL1D_KERNEL_SIZE = 3
GRU_HIDDEN_SIZE = 200
FULLY_CONNECTED_LAYER_SIZE = 32

NUMBER_EPOCHS = 20
LEARNING_RATE = 0.0001

#LOSS_WEIGHT_POSITIVE = 0.099
#LOSS_WEIGHT_NEGATIVE = 0.901

LOSS_WEIGHT_POSITIVE = 1
LOSS_WEIGHT_NEGATIVE = 1

################################################
def load_text_file(file_text):
    start_time = time.time()
    with open(file_text) as f:
        lines = f.readlines()
    #print "Loading time: ", time.time() - start_time
    #print "nb lines: ", len(lines)
    return lines

def load_lst_file(file_lst, dir_name):
    lst_file = load_text_file(file_lst)
    lst_path = [dir_name + file_name.strip() + '.pssm' for file_name in lst_file]
    return lst_path

class BioinformaticsDataset(Dataset):
    def __init__(self, positive_lst_file, negative_lst_file, positive_dir, negative_dir):
        lst_path_positive = load_lst_file(positive_lst_file, positive_dir)
        print("positive: ", len(lst_path_positive))
        lst_path_negative = load_lst_file(negative_lst_file, negative_dir)
        print ("negative: ", len(lst_path_negative))

        self.lst_path = lst_path_negative + lst_path_positive
        self.nb_negative = len(lst_path_negative)

    def __getitem__(self, index):
        print("\nindex: ", index, " ==> ", self.lst_path[index])
        label = 1
        if index < self.nb_negative :  label = 0

        weight = LOSS_WEIGHT_POSITIVE
        if label == 0: weight = LOSS_WEIGHT_NEGATIVE

        # print(" label: ", label)

        lines = load_text_file(self.lst_path[index])
        # print("lines: ", len(lines))
        start_line = 3
        end_line = len(lines) - 7

        # print(start_line, " : ", end_line)

        values = np.zeros((end_line - start_line + 1, 20))

        for i in range(start_line, end_line + 1):
            #print i
            strs = lines[i].strip().split()[2:22]
            #print strs
            for j in range(20):
                values[i-start_line][j] = int(strs[j])
                #print(values[i-start_line][j])
            #break

        # print("values: ", values)

        input = torch.from_numpy(values)
        return input,label, weight

    def __len__(self):
        return len(self.lst_path)

class RNNModel(nn.Module):
    def __init__(self, n_layers=1):
        super(RNNModel, self).__init__()
        self.c1 = nn.Conv1d(20, CONV1D_FEATURE_SIZE, CONV1D_KERNEL_SIZE)
        self.p1 = nn.AvgPool1d(AVGPOOL1D_KERNEL_SIZE)
        self.c2 = nn.Conv1d(CONV1D_FEATURE_SIZE, CONV1D_FEATURE_SIZE, CONV1D_KERNEL_SIZE)
        self.p2 = nn.AvgPool1d(AVGPOOL1D_KERNEL_SIZE)
        self.gru = nn.GRU(CONV1D_FEATURE_SIZE, GRU_HIDDEN_SIZE, n_layers, dropout=0.01)
        self.fc = nn.Linear(GRU_HIDDEN_SIZE, FULLY_CONNECTED_LAYER_SIZE)
        self.fc_drop = nn.Dropout()
        self.out = nn.Linear(FULLY_CONNECTED_LAYER_SIZE, 1)
        self.out_act = nn.Sigmoid()

        self.gru_layers = n_layers

    def forward(self, inputs):
        batch_size = inputs.size(0)
        #print "inputs size: ", inputs.size()
        #print "batch_size: ", batch_size
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.gru_layers, inputs.size(0), GRU_HIDDEN_SIZE).cuda())
        else:
            h0 = Variable(torch.zeros(self.gru_layers, inputs.size(0), GRU_HIDDEN_SIZE))
        #print h0.size()

        # Turn (batch_size x seq_len) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(1,2)
        #print "inputs size: ", inputs.size()
        c = self.c1(inputs)
        #print "c1 : ", c.size()
        p = self.p1(c)
        #print "p1: ", p.size()
        c = self.c2(p)
        #print "c2: ", c.size()
        p = self.p2(c)
        #print "p2: ", p.size()

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)

        #p = F.tanh(p)
        p = F.relu(p)
        #print "p: ", p.size()

        #output, hidden = self.gru(p, hidden)
        output, hidden = self.gru(p, h0)
        #conv_seq_len = output.size(0)
        #print "gru output: ", output.size()
        #print "conv_seq_len: ", conv_seq_len
        #print "batch_size: ", batch_size
        #print "GRU_HIDDEN_SIZE: ", GRU_HIDDEN_SIZE

        #output = output.view(conv_seq_len * batch_size, GRU_HIDDEN_SIZE)
        #print "last gru output: ", output.size()
        #output = torch.gather(output, 1, torch.tensor(30))
        #print "last gru output: ", output.size()

        #output = F.tanh(self.out(output))
        #print("Hidden: ", hidden.size())

        output = F.relu(self.fc(hidden))
        #print("output fc: ", output.size())
        output = self.out(output)
        #print("output out: ", output.size())
        output = self.out_act(output)
        #print("output sigmoid: ", output.size())
        #output = output.view(conv_seq_len, -1, NUMBER_CLASSES)

        return output

def train_one_epoch(learning_rate):
    model.train()
    # Dieu chinh learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    epoch_loss_train = 0.0
    nb_train = 0
    nb_train_short = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels, weight = data
        #print("inputs: ", inputs.size())
        inputs_length = inputs.size()[1]
        #print("inputs_length: ", inputs_length)
        if inputs_length < 20:
            nb_train_short += 1
            continue

        inputs = inputs.float()
        labels = labels.float()
        weight = weight.float()

        #print "\nBatch i : ", i
        # print "inputs: ", inputs
        #print "labels: ", labels

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            weight = Variable(weight.cuda(), requires_grad=False)
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            weight = Variable(weight, requires_grad=False)

        # print(epoch, i, "inputs", inputs.data, "labels", labels.data)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(inputs)
        outputs = outputs[-1,-1]

        #loss = criterion(outputs[-1], labels)
        #print("outputs: ", outputs.size())
        #print("labels: ", labels.size())
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        #print("loss: ", loss.data[0])
        loss = torch.mul(loss, weight)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        epoch_loss_train = epoch_loss_train + loss.data[0]
        nb_train = nb_train + 1

        #if i > 10: break
        #if i % 1000 == 0:
        #    print("Loss: ", i, " : ", epoch_loss_train/nb_train)

    print("nb train short (length < 20): ", nb_train_short)

    epoch_loss_train_avg = epoch_loss_train / nb_train
    return epoch_loss_train_avg

def evaluate(loader, is_val=True):
    model.eval()

    epoch_loss = 0.0
    nb = 0
    nb_short = 0

    arr_labels = []
    arr_labels_hyp = []
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels, weight = data
        #print "labels: ", labels
        # print("inputs: ", inputs.size())
        inputs_length = inputs.size()[1]
        # print("inputs_length: ", inputs_length)
        if inputs_length < 20:
            nb_short += 1
            continue

        #print "labels: ", labels.cpu().numpy()[0]
        arr_labels.append(labels.cpu().numpy()[0])

        inputs = inputs.float()
        labels = labels.float()

        # print "\nBatch i : ", i
        # print "inputs: ", inputs
        # print "labels: ", labels

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # print(epoch, i, "inputs", inputs.data, "labels", labels.data)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(inputs)
        outputs = outputs[-1, -1]

        # loss = criterion(outputs[-1], labels)
        # print("outputs: ", outputs.size())
        # print("labels: ", labels.size())
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)

        #print("outputs: ", outputs)
        if outputs.data.cpu().numpy()[0] > 0.5: arr_labels_hyp.append(1)
        else: arr_labels_hyp.append(0)

        epoch_loss = epoch_loss + loss.data[0]
        nb = nb + 1

        #if i > 10: break
        # if i % 1000 == 0:
        #    print("Loss: ", i, " : ", epoch_loss_train/nb_train)

    #print("nb short (length < 20): ", nb_short)

    # print("arr_labels: ", arr_labels)
    # print("arr_labels_hyp: ", arr_labels_hyp)
    epoch_loss_avg = epoch_loss / nb

    if is_val:
        acc = calculate_accuracy(arr_labels, arr_labels_hyp)
        return epoch_loss_avg, acc
    else:
        acc, confusion_matrix, sensitivity, specificity, labs, preds = calculate_confusion_matrix(arr_labels, arr_labels_hyp)
        return epoch_loss_avg, acc, confusion_matrix, sensitivity, specificity, labs, preds

def calculate_accuracy(arr_labels, arr_labels_hyp):
    corrects = 0

    for i in range(len(arr_labels)):
        if arr_labels[i] == arr_labels_hyp[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(arr_labels)
    return acc

def calculate_confusion_matrix(arr_labels, arr_labels_hyp):
    corrects = 0
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(arr_labels)):
        confusion_matrix[arr_labels_hyp[i]][arr_labels[i]] += 1

        if arr_labels[i] == arr_labels_hyp[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(arr_labels)
    sensitivity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    specificity = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    return acc, confusion_matrix, sensitivity, specificity, arr_labels, arr_labels_hyp

def validation():
    model.load_state_dict(torch.load(file_model))
    model.eval()

    test_set = BioinformaticsDataset(val_positive_lst_file, val_negative_lst_file,
                                     val_positive_dir, val_negative_dir)
    print("Length Test Dataset: ", len(test_set.lst_path), "\n")

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             # batch_size=32,
                             shuffle=True,
                             num_workers=4)

    epoch_loss_test_avg, test_acc, confusion_matrix, sensitivity, specificity, labs, preds = evaluate(test_loader, is_val=False)
    print("Validation loss: ", epoch_loss_test_avg)
    print("Validation acc: ", test_acc)
    print("Validation confusion matrix: \n", confusion_matrix)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("Labels: ", labs)
    print("Predict: ", preds)

def test():
    model.load_state_dict(torch.load(file_model))
    model.eval()

    test_set = BioinformaticsDataset(test_positive_lst_file, test_negative_lst_file,
                                     test_positive_dir, test_negative_dir)
    print("Length Test Dataset: ", len(test_set.lst_path), "\n")

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             # batch_size=32,
                             shuffle=True,
                             num_workers=4)

    epoch_loss_test_avg, test_acc, confusion_matrix, sensitivity, specificity, labs, preds = evaluate(test_loader, is_val=False)
    print("Test loss: ", epoch_loss_test_avg)
    print("Test acc: ", test_acc)
    print("Test confusion matrix: \n", confusion_matrix)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("Labels: ", labs)
    print("Predict: ", preds)

def train():
    train_losses = []
    val_losses = []
    val_acc = []

    best_val_loss = 1000

    for epoch in range(NUMBER_EPOCHS):
        epoch_loss_train_avg = train_one_epoch(learning_rate)
        train_losses.append(epoch_loss_train_avg)

        epoch_loss_val_avg, acc = evaluate(val_loader)
        val_losses.append(epoch_loss_val_avg)
        val_acc.append(acc)

        if best_val_loss > epoch_loss_val_avg:
            torch.save(model.state_dict(), file_model)
            best_val_loss = epoch_loss_val_avg
            print("Save model, best_val_loss: ", best_val_loss)

    print("train_losses: ", train_losses)
    print("val_losses: ", val_losses)
    print("val_acc: ", val_acc)

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr
############### MAIN PROGRAM ##############
if __name__ == "__main__":
    train_set = BioinformaticsDataset(train_positive_lst_file, train_negative_lst_file,
                                      train_positive_dir, train_negative_dir)
    print("Length Train Dataset: ", len(train_set.lst_path), "\n")

    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,
                              # batch_size=32,
                              shuffle=True,
                              num_workers=4)

    val_set = BioinformaticsDataset(val_positive_lst_file, val_negative_lst_file,
                                      val_positive_dir, val_negative_dir)
    print("Length Validation Dataset: ", len(val_set.lst_path), "\n")

    val_loader = DataLoader(dataset=val_set,
                              batch_size=1,
                              # batch_size=32,
                              shuffle=True,
                              num_workers=4)

    model = RNNModel(1)
    # model.load_state_dict(torch.load("save15.model"))
    print(model)
    print(torch_summarize(model))

    if torch.cuda.is_available():
        model.cuda()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()

    learning_rate = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train()

    validation()

    # test()