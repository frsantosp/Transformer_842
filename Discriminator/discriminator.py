import torch
import torch.nn as nn
import random

class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, batch_first=True):
        super(SentenceClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden.squeeze(0))
        return out


import torch.nn.functional as F

def generator_loss(gen_output, target_output, sentence_classifier):
    # Calculate the original KLDivLoss
    kldiv_loss = F.kl_div(gen_output, target_output)
    
    # Calculate the additional penalty
    classifier_output = sentence_classifier(gen_output)
    penalty = F.binary_cross_entropy_with_logits(classifier_output, torch.zeros_like(classifier_output))
    
    # Combine the losses
    loss = kldiv_loss + penalty
    return loss

def train_sentence_classifier(model, dataset, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(dataset)):
            # Get a real German sentence from the dataset
            data = dataset[i]
            real_sentence = data['german']
            
            # Generate a fake German sentence by randomly sampling words from the vocabulary
            fake_sentence_length = random.randint(dataset.german_min_len, dataset.german_max_len)
            fake_sentence = torch.randint(0, dataset.german_vocab_len, (fake_sentence_length,))
            
            # Concatenate the real and fake sentences and create labels
            sentences = torch.cat((real_sentence.unsqueeze(0), fake_sentence.unsqueeze(0)), dim=0)
            labels = torch.tensor([1., 0.])
            
            # Forward pass
            optimizer.zero_grad()
            output = model(sentences)
            loss = criterion(output.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(dataset)}')