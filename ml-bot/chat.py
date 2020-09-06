import torch
import numpy as np
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import training
from training import Net, token
import random

while 1:
    arr = np.zeros(len(training.words), dtype=np.float32)

    inp = str(input("you: "))
    
    if inp == "exit":
        break

    inp = token(inp)

    for i in range(len(arr)):
        for j in range(len(inp)):
            if inp[j] == training.words[i]:
                arr[i]=1.0

    arr = arr.reshape(1, arr.shape[0])

    # convert numpy array to tensor
    t = torch.from_numpy(arr)
    
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")              

    # load pre-trained model
    d = torch.load('model.pth')
    net = Net()
    net.load_state_dict(d)
    net.eval()
    out = net(t).to(device)
    _, predict = torch.max(out, dim=1)
    tag = training.tags[predict.item()]
    for intent in training.intents['intents']:
        if tag == intent['tag']:
            print("bot:", random.choice(intent['responses']))
