"""
Seq2Seq using Transformers on the Multi30k
dataset. In this video I utilize Pytorch
inbuilt Transformer modules, and have a
separate implementation for Transformers
from scratch. Training this model for a
while (not too long) gives a BLEU score
of ~35, and I think training for longer
would give even better results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import argparse
"""
Command Line Arguments
"""
parser = argparse.ArgumentParser(description='Argument Parser for optimizer, dropout, activation function, embedding matrix, number of epochs, and learning rate.')

parser.add_argument('--epochs', type=int, required=False, help='How many epochs to train for', default=100)
parser.add_argument('--CUDA', type=str, required=False, choices=["True", "False"], help='Train on gpu', default="True")
parser.add_argument('--load', type=str, required=False, choices=["True", "False"], help='Load model from checkpoint', default="True")
parser.add_argument('--save_interval', type=int, required=False, help='How many epochs to save the model', default=10)
parser.add_argument('--loss_save_path', type=str, required=False,  help='directory to save loss data to', default="Loss")
parser.add_argument('--num_heads', type=int, help="Number of heads for multi-headed attention", default=8)
parser.add_argument('--batch_size', type=int, help="Batch_size", default=32)
parser.add_argument('--embed_dim', type=int, help="Size of token embeddings", default=512)
parser.add_argument('--print_interval', type=int, help="Interval on which to print and validate data", default=100)
parser.add_argument('--learning_rate', type=float, help="Learning rate for backprop", default=6e-4)
parser.add_argument('--layers', type=int, help="number of layer for encoder and decoder", default=3)
parser.add_argument('--dropout', type=float, help="Dropout rate for transformer", default=0.1)
parser.add_argument('--file_name', type=str, help="Name of checkpoint file", default="transformer_base.pth.tar")

args = parser.parse_args()

print("Using arguments: ", args)


"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = args.load == "True"
save_model = True

# Training hyperparameters
num_epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = args.embed_dim
num_heads = args.num_heads
num_encoder_layers = args.layers
num_decoder_layers = args.layers
dropout = args.dropout
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]
filename = args.file_name

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step_train= 0
step_valid = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load(filename), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=filename)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )
    valid_lossed = []
    for batch_idx, batch in enumerate(valid_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        valid_lossed.append(loss.item())

        # plot to tensorboard
        writer.add_scalar("Validation loss", loss, global_step=step_valid)
        step_valid += 1

    print(f"Translated example sentence: \n {translated_sentence}")
    mean_loss = sum(valid_lossed) / len(valid_lossed)
    print(f"Mean loss for epoch {epoch + 1}: {mean_loss}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step_train)
        step_train += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")