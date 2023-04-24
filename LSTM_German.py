#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import spacy
import random
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


# In[9]:


# Load German and English tokenizers
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Tokenization functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_vocab(data, tokenizer, min_freq=2):
    return build_vocab_from_iterator((tokenizer(text) for text in data), min_freq=min_freq)

# Get dataset and split
train_and_valid_data, test_data = Multi30k(split=('train', 'test'), language_pair=('de', 'en'))

# Convert dataset to list
train_and_valid_data = list(train_and_valid_data)
test_data = list(test_data)

# Calculate the number of samples for train and validation
n_train = int(len(train_and_valid_data) * 0.9)
n_valid = len(train_and_valid_data) - n_train

# Split the train and valid data
train_data, valid_data = random_split(train_and_valid_data, [n_train, n_valid])

# Build vocab
SRC = build_vocab([src for src, _ in train_data], tokenize_de)
TRG = build_vocab([trg for _, trg in train_data], tokenize_en)


# In[10]:


# Custom collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = torch.tensor([len(src) for src in src_batch])
    trg_lengths = torch.tensor([len(trg) for trg in trg_batch])
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=SRC['<pad>'])
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=TRG['<pad>'])
    return src_padded, trg_padded

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=64, collate_fn=collate_fn)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[11]:


# Define the LSTM-based Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell =self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


# In[12]:


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)


# In[27]:


optimizer = optim.Adam(model.parameters())
#TRG_PAD_IDX = TRG.vocab.get_stoi['<pad>',None]
criterion = nn.CrossEntropyLoss()


# In[28]:


TRG.vocab.get_stoi


# In[29]:


# Training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Training loop
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')


# In[30]:


for epoch in range(N_EPOCHS):
    train_iterator = Iterator(train_data, batch_size=64, device=device)
    valid_iterator = Iterator(valid_data, batch_size=64, device=device)

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstmfinalmodel.pt')


# In[33]:


#from torchtext.data import Field, TabularDataset
from torchtext.data import Iterator


# In[40]:


def create_batches(data, batch_size, src_field, trg_field, device):
    data = list(data)  # Convert the data to a list
    random.shuffle(data)  # Shuffle the data for each epoch
    data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    batch_list = []
    
    for batch_data in data_batches:
        src_list = [src_field.process([datum.src], device=device) for datum in batch_data]
        trg_list = [trg_field.process([datum.trg], device=device) for datum in batch_data]
        
        src_list, trg_list = zip(*sorted(zip(src_list, trg_list), key=lambda x: len(x[0]), reverse=True))
        
        src_list = torch.cat(src_list)
        trg_list = torch.cat(trg_list)
        
        batch_list.append((src_list, trg_list))
    
    return batch_list


# In[41]:


train_batches = create_batches(train_data, batch_size=64, src_field=SRC, trg_field=TRG, device=device)
valid_batches = create_batches(valid_data, batch_size=64, src_field=SRC, trg_field=TRG, device=device)


# In[44]:


import torch
import fairseq

# Load the pre-trained model
model = fairseq.models.transformer.TransformerModel.from_pretrained(
    'transformer.wmt19.en-de',
    checkpoint_file='model1.pt',
    data_name_or_path='wmt14.en-de'
)

# Set the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to evaluation mode
model.eval()

# Define the German tokenizer
de_tokenizer = fairseq.data.encoders.build_tokenizer('moses', source_lang='de')

# Define the English tokenizer
en_tokenizer = fairseq.data.encoders.build_tokenizer('moses', source_lang='en')

# Define the German to English translation function
def translate_de_to_en(model, sentence, de_tokenizer, en_tokenizer, device):
    # Tokenize the German sentence
    sentence = de_tokenizer.encode(sentence)
    # Convert the tokenized sentence to a tensor
    sentence = torch.LongTensor(sentence).unsqueeze(0).to(device)
    # Generate the English translation
    translation = model.translate(sentence)
    # Detokenize the English translation
    translation = en_tokenizer.decode(translation[0]['tokens'])
    return translation

# Example usage
german_sentence = 'Das ist ein Beispiel.'
english_translation = translate_de_to_en(model, german_sentence, de_tokenizer, en_tokenizer, device)
print(english_translation)


# In[ ]:





# In[45]:


import spacy
import string
import nltk
from collections import Counter

# Tokenizer
nlp = spacy.load('de_core_news_sm')
def tokenize_de(text):
    return [tok.text for tok in nlp(text)]

# Vocabulary
def build_vocab(texts, max_vocab_size=None):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_de(text))
    if max_vocab_size is not None:
        counter = dict(counter.most_common(max_vocab_size))
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for token in counter:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# Example usage
german_sentences = ['ein pferd geht unter einer brücke neben einem boot',
                    'ein mann sitzt vor einem café und liest eine zeitung']
vocab = build_vocab(german_sentences, max_vocab_size=10000)
print(vocab)


# In[46]:


import torch

# Dictionary that maps tokens to indices
token_to_idx = {token: idx for token, idx in vocab.items()}

# Example sequence
sequence = 'ein pferd geht unter einer brücke neben einem boot'
tokenized_sequence = tokenize_de(sequence)
indexed_sequence = torch.LongTensor([token_to_idx.get(token, token_to_idx['<unk>']) for token in tokenized_sequence])
print(indexed_sequence)


# In[ ]:





# In[ ]:





# In[53]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define input and output language
INPUT_LANG = "de"
OUTPUT_LANG = "en"

# Define the input and output sequences
input_seq = ["Ich bin ein Berliner", "Ich liebe dich"]
output_seq = ["I am a Berliner", "I love you"]

# Define vocabulary and initialize tokens
input_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
output_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
input_token_count = 3
output_token_count = 3

# Tokenize input and output sequences
for sentence in input_seq:
    for token in sentence.split():
        if token not in input_vocab:
            input_vocab[token] = input_token_count
            input_token_count += 1

for sentence in output_seq:
    for token in sentence.split():
        if token not in output_vocab:
            output_vocab[token] = output_token_count
            output_token_count += 1

# Define the encoder
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input_seq, hidden_state):
        output, hidden = self.lstm(input_seq, hidden_state)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Define the decoder
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        output, hidden = self.lstm(input_seq, hidden_state)
        output = self.linear(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# In[54]:


# Define the training loop
def train(encoder, decoder, input_tensor, output_tensor, criterion, optimizer, max_length=20):
    # Initialize the hidden state of the encoder
    encoder_hidden = encoder.init_hidden(input_tensor.size(1))

    # Zero the gradients
    optimizer.zero_grad()

    # Initialize the loss
    loss = 0

    # Encode the input sequence
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    # Initialize the hidden state of the encoder
    encoder_hidden = encoder.init_hidden(input_tensor.size(1))

    # Initialize the decoder input
    decoder_input = torch.tensor([[output_vocab["<SOS>"]] * input_tensor.size(1)])

    # Initialize the decoder hidden state with the last hidden state of the encoder
    decoder_hidden = encoder_hidden

    # Teacher forcing: use the true output sequence as the input for the decoder
    for i in range(output_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, output_tensor[i])
        decoder_input = output_tensor[i].unsqueeze(0)  # Add a dimension to match the decoder_input tensor shape

    # Backpropagate the error and update the parameters
    loss.backward()
    optimizer.step()

    return loss.item() / output_tensor.size(0)

# Initialize the hyperparameters
hidden_size = 256
learning_rate = 0.01
num_epochs = 100

# Define the encoder and decoder
encoder = EncoderLSTM(input_token_count, hidden_size)
decoder = DecoderLSTM(output_token_count, hidden_size, output_token_count)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(len(input_seq)):
        # Convert the input and output sequences into tensors
        input_tensor = torch.tensor([input_vocab[token] for token in input_seq[i].split()]).unsqueeze(0).transpose(0, 1)
        output_tensor = torch.tensor([output_vocab[token] for token in output_seq[i].split()]).unsqueeze(0).transpose(0, 1)

        # Train the model on the current example
        loss = train(encoder, decoder, input_tensor, output_tensor, criterion, optimizer)

        # Update the epoch loss
        epoch_loss += loss

    # Compute the average loss for the epoch
    epoch_loss /= len(input_seq)

    # Print the epoch loss
    print("Epoch {} loss: {:.4f}".format(epoch+1, epoch_loss))



# In[ ]:





# In[ ]:





# In[ ]:





# In[50]:


# Test the model on a new sentence
with torch.no_grad():
    input_sentence = "Ich bin hungrig"
    input_tensor = torch.tensor([input_vocab[token] for token in input_sentence.split()]).unsqueeze(0).transpose(0, 1)
    encoder_hidden = encoder.init_hidden(1)
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.tensor([[output_vocab["<SOS>"]]])
    decoder_hidden = encoder_hidden
    output_sentence = ""
    for i in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        if topi.item() == output_vocab["<EOS>"]:
            break
        else:
            output_sentence += " " + list(output_vocab.keys())[list(output_vocab.values()).index(topi.item())]
        decoder_input = topi.squeeze(1)
    print("Input sentence:", input_sentence)
    print("Output sentence:", output_sentence.strip())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




