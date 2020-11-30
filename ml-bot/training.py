import torch
import numpy as np
import os
import json
import nltk
import tqdm
from nltk.stem.lancaster import LancasterStemmer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader 

# model save path
path = "model.pth"

stemmer = LancasterStemmer()
data = open('intents.json', 'r')
intents = json.load(data)

words = []
tags = []
xy = []

def token(string):
    return nltk.word_tokenize(string)

data = open('intents.json', 'r')
intents = json.load(data)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = token(pattern)
        words.extend(word)
        xy.append((word, tag))
     
words = [stemmer.stem(w.lower()) for w in words]        

words = sorted(list(set(words)))
tags = sorted(set(tags))

x_train = [] # input
y_train = [] # output

# i variable is tokenized strings, tag variable is the strings tag in intents.json
for (i, tag) in xy:
    i = [stemmer.stem(w) for w in i]
    arr = np.zeros(len(words), dtype=np.float32)  
    for j in range(len(arr)):
        for x in range(len(i)):
            if i[x] == words[j]:
                arr[j]=1.0
    x_train.append(arr)
    y_train.append(tags.index(tag))  

# convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# dataset class for nn
class dataset(Dataset):
    def __init__(self):
        self.samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.samples

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(arr), 32) # input layer
        self.fc2 = nn.Linear(32, 32) # hidden layer
        self.fc3 = nn.Linear(32, len(tags)) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    net = Net().to(device)

    batch_size = 16
    ds = dataset()
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    epochs = 20
    print("training model", epochs, "times")
    for epoch in range(epochs):
        for (words, labels) in loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = net(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    try:
        torch.save(net.state_dict(), path)
        print("pytorch model saved in:", path)
    except:
        print("failed to save pytorch model at:", path)
    
