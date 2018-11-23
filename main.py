import argparse
import time
import math

import torch.nn as nn
import torch
import torch.optim as optim

from model import Net
from readData import ManageData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(net, data, criterion, optimizer):
    total_data = len(data.x1_train)
    total_loss = 0
    i = 0
    for x1,x2,target in data.getData():#data.getDataTrain():

        printProgressBar(i, total_data, prefix = 'Progress:', suffix = 'Complete', length = 50)

        x1 = x1.to(device)
        x2 = x2.to(device)
        target = target.to(device)

        output = net(x1, x2)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        i += 1

    return total_loss/total_data

def test(data, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for x1,x2,target in data.getData(False):#data.getDataTest():
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)

            outputs = net(x1, x2)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy in test: %d %%' % (
        100 * correct / total))

    return (100 * correct / total)

def finalTest(data, net):
    f = open("results.csv", "w")
    f.write("Id,Category\n")
    with torch.no_grad():
        for x1,x2,t_id in data.getDataTestFinal():
            x1 = x1.to(device)
            x2 = x2.to(device)

            outputs = net(x1, x2)

            if outputs.item() == 0:
                label = 'agreed'
            if outputs.item() == 1:
                label = 'disagreed'
            if outputs.item() == 2:
                label = 'unrelated'
            f.write(str(t_id) + ',' + label + '\n')

def main(args):
    n_iters = 50
    print_every = 5
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

    data = ManageData(args['path'])
    print("Data loaded")
    net = Net()
    net.to(device)
    print("Model Created")

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        adjust_learning_rate(optimizer, iter, learning_rate)
        loss = train(net, data, criterion, optimizer)
        acc = test(data, net)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            print('%d %d%% (%s) %.4f' % (iter, iter / n_iters * 100, timeSince(start), loss))

    finalTest(data, net)

def getArgsCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',default='./data', help='path to dataset')
    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = getArgsCommand()
    main(args)