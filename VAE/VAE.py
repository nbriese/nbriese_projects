# Variational Auto Encoder
# Nathan Briese

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import getopt

def main(argv):
    learn_rate = 0.01
    num_epochs = 10
    train_switch = False

    try:
        opts, args = getopt.getopt(argv,"tl:e:",["lr=", "e="])
    except getopt.GetoptError:
        print("usage: python3 VAE.py [-t | -l learning rate | -e num epochs]")
        print("Options and arguments:")
        print("-t train a new model")
        print("-l specify the learning rate")
        print("-e specify the number of epochs")
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-t':
            train_switch = True
        elif opt == '-l':
            learn_rate = float(arg)
        elif opt == '-e':
            num_epochs = int(arg)

    if(train_switch):
        torch.set_default_tensor_type(cuda.FloatTensor)
        train(num_epochs, learn_rate)
    test()

class VAE_NET(nn.Module):
    def __init__(self):
        super(VAE_NET, self).__init__()
        self.fc1  = nn.Linear(784, 400)
        self.fc2  = nn.Linear(400, 50)
        self.fc3  = nn.Linear(50, 400)
        self.fc4  = nn.Linear(400, 784)
        self.sig  = nn.Sigmoid()

    def encode(self, x):
        # Input: training image, size 28x28
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        u = self.fc2(x)
        s = self.fc2(x)
        # e is unit normal noise
        e = torch.randn(u.size())
        z = u + (s * e)
        # Output: compressed version of image, size 20
        return z

    def decode(self, z):
        # Input: compressed image, size 20
        z = F.relu(self.fc3(z))
        z = self.sig(self.fc4(z))
        # Output: denoised original image, size 784
        return z

    def forward(self, x):
        return self.decode(self.encode(x))

def train(num_epochs, learn_rate):
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    net = VAE_NET().cuda()

    criterion = nn.BCELoss(reduction='sum').cuda()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    loss_list = np.zeros(num_epochs)

    print("Beginning Training with %d epochs" % num_epochs)
    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1)
        running_loss = 0.0
        for _, (inputs, _) in enumerate(trainloader, 0):
            inputs = inputs.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward and backward and optimize
            outputs = net(inputs)
            # the target is the original image
            loss = criterion(outputs, inputs.view(-1, 784))
            loss.backward()
            optimizer.step()

            # aggregate loss
            # running_loss += loss.item()*inputs.size()[0]

        # loss_list[epoch] = running_loss/len(trainloader)

    # Save the model
    torch.save(net.state_dict(), './vae.pth')
    print('Finished Training')

def test():
    # generate images similar to MNIST dataset
    # Show a grid
    test_net = VAE_NET()
    test_net.load_state_dict(torch.load('./vae.pth'))

    # randomly generate 16 new Z from unit Gaussian
    inputs = torch.randn(16, 50)
    images = test_net.decode(inputs)

    plt.figure()
    plt.suptitle("Recreating Images from Random Noise")
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(torch.reshape(images[i], (28,28)).detach().numpy(), cmap='gray') # , vmin=0.0, vmax=1.0)
    plt.savefig("./VAE_output.png")
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
