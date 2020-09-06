# ml-bot
a machine-learning chatbot
# model
the model is a feedforward neural-network that is already trained but you can always train the neural-network on your own if you'd like that. the model is saved in model.pth. The input to the model is a numpy array consisting of 1's and 0's. 1 if a word is in your sentence, 0 if it's not. The numpy array has the same size/length as the list with all of the words that the model can identify. 
# intents
All of the words, input examples and responses is inside of the intents.json file. You can always modify this file to give the bot more functions, but then you'll also have to train it again to get the correct responses. 

Write exit to break the loop
