from Dataset.translation_dataset import EnglishToGermanDataset
from Transformer.transfomer import TransformerTranslator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from Discriminator.discriminator import SentenceClassifier
from torch.nn.functional import binary_cross_entropy_with_logits
import argparse

"""
Command Line Arguments
"""
parser = argparse.ArgumentParser(description='Argument Parser for optimizer, dropout, activation function, embedding matrix, number of epochs, and learning rate.')

parser.add_argument('--epochs', type=int, required=False, help='How many epochs to train for', default=1000)
parser.add_argument('--LAMBDA', type=float, required=False, help='weight of penalty term in loss function', default=0.01)
parser.add_argument('--CUDA', type=str, required=False, choices=["True", "False"], help='Train on gpu', default="True")

args = parser.parse_args()

"""
Hyperparameters
"""
CUDA = (args.CUDA == "True")
PRINT_INTERVAL = 5000
VALIDATE_AMOUNT = 10
SAVE_INTERVAL = 5000
LAMBDA = args.LAMBDA

batch_size = 128
embed_dim = 64
num_blocks = 2
num_heads = 1  # Must be factor of token size
max_context_length = 1000

num_epochs = args.epochs
learning_rate = 1e-3

use_teacher_forcing = False

device = torch.device("cuda" if CUDA else "cpu")

"""
Dataset
"""
dataset = EnglishToGermanDataset(CUDA=CUDA)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
dataloader_test = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device)
)

"""
Model
"""
encoder_vocab_size = dataset.english_vocab_len
output_vocab_size = dataset.german_vocab_len
torch.set_default_tensor_type(torch.cuda.FloatTensor if CUDA else torch.FloatTensor)
model = TransformerTranslator(
    embed_dim, num_blocks, num_heads, encoder_vocab_size,output_vocab_size,CUDA=CUDA
).to(device)


"""
Discriminator Model
"""
sentence_classifier = SentenceClassifier(
    vocab_size=dataset.german_vocab_len,
    embedding_dim=embed_dim,
    hidden_dim=128,
    output_dim=1
).to(device)

classifier_optimizer = torch.optim.Adam(sentence_classifier.parameters(), lr=1e-3)
classifier_criterion = nn.BCEWithLogitsLoss()

"""
Loss Function + Optimizer
"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(reduction='batchmean')

"""
Load From Checkpoint
"""
LOAD = -1

if LOAD != -1:
    checkpoint = torch.load(
        os.path.join("Checkpoints", "Checkpoint" + str(LOAD) + ".pkl")
    )
    test_losses = checkpoint["test_losses"]
    train_losses = checkpoint["train_losses"]
    num_steps = checkpoint["num_steps"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    test_losses = []
    train_losses = []
    num_steps = 0
"""
Train Loop
"""
for epoch in range(num_epochs):
    running_loss = []
    classifier_running_loss = []
    running_test_loss = []
    dataset.train()
    """
    TRAIN LOOP
    """
    for idx, item in enumerate(dataloader):
        """
        ============================================================
        """
        model.train()
        sentence_classifier.train()

        ###################
        # Zero Gradients
        model.zero_grad()
        optimizer.zero_grad()
        ###################

        ###################
        # Encode English Sentence
        model.encode(item["english"])
        ###################

        ###################
        # Output German, One Token At A Time
        all_outs = torch.tensor([], requires_grad=True).to(device)
        all_outs_tokens = item["german"][:,:1]
        for i in range(item["german"].shape[1]-1):  
            if(use_teacher_forcing):
                out = model(item["german"][:,: i+1])
            else:
                #Minus one because we don't need to predict from <end> token
                out = model(all_outs_tokens[:, : i + 1]) #Start with seeing <start> token
            all_outs = torch.cat((all_outs, out), dim=1)
            #Get the token output, so to feed it back to self in next round.
            out_token = torch.argmax(out,dim=1)
            all_outs_tokens = torch.cat((all_outs_tokens,out_token),dim=1)
        ###################
        
        ###################
        # Mask Out Extra Padded Tokens In The End(Optional)
        all_outs = all_outs * item["logit_mask"][:, 1:, :]
        item["logits"] = item["logits"] * item["logit_mask"]
        ###################
    

        ###################
        # BackProp
        # Calculate the additional penalty
        predicted_indices = torch.argmax(all_outs, dim=-1)
        # print(item["logits"])
        # print(item["logits"].shape)
        # print(all_outs)
        # print(predicted_indices)
        classifier_output = sentence_classifier(predicted_indices.detach())
        penalty = binary_cross_entropy_with_logits(classifier_output, torch.zeros_like(classifier_output))

        loss = criterion(all_outs, item["logits"][:, 1:, :]) + LAMBDA*penalty

        loss.backward()
        optimizer.step()
        ###################

        ###################
        # Train the SentenceClassifier
        ###################

        # Create a batch of real German sentences
        real_sentences = item["german"][:, :-1] #don't include eos
        #real_sentences = item["german"][:, 1:] #may need to do this so we don't include <start>

        # Generate a batch of fake German sentences by sampling words from the vocabulary
        fake_sentences = torch.argmax(all_outs.detach(), dim=2)

        # Concatenate the real and fake sentences and create labels
        sentences = torch.cat((real_sentences, fake_sentences), dim=0)
        if idx != len(dataloader) - 1:
            labels = torch.tensor([1.]*batch_size + [0.]*batch_size).to(device)
        else:
            labels = torch.tensor([1.]*len(real_sentences) + [0.]*len(fake_sentences)).to(device)

        # Forward pass
        classifier_optimizer.zero_grad()
        # print(sentences.shape, "sents")
        output = sentence_classifier(sentences)
        # print(output.shape)
        # print(labels.shape)
        classifier_loss = classifier_criterion(output.squeeze(), labels)

        # Backward pass
        classifier_loss.backward()
        classifier_optimizer.step()

        classifier_running_loss.append(classifier_loss.item())

        running_loss.append(loss.item())
        num_steps += 1
        """
        ============================================================
        """
        if num_steps % PRINT_INTERVAL == 0 or idx == len(dataloader) - 1:
            """
            Validation LOOP
            """
            all_outs.detach().cpu()
            item["logits"].detach().cpu()
            dataset.test()
            model.eval()
            sentence_classifier.eval()
            running_classifier_test_loss = []
            with torch.no_grad():
                for jdx, item in enumerate(dataloader_test):
                    model.encode(item["english"])
                    all_outs = torch.tensor([], requires_grad=False).to(device)
                    all_outs_tokens = item["german"][:,:1]
                    for i in range(item["german"].shape[1] - 1):
                        #No teacher forcing in validation                        
                        out = model(all_outs_tokens[:,:i+1])
                        out_token = torch.argmax(out,dim=1)
                        all_outs = torch.cat((all_outs, out), dim=1)                        
                        all_outs_tokens = torch.cat((all_outs_tokens,out_token),dim=1)
                    all_outs = all_outs * item["logit_mask"][:,1:,:]
                    item["logits"] = item["logits"] * item["logit_mask"]
                    loss = criterion(all_outs, item["logits"][:,1:,:])
                    running_test_loss.append(loss.item())

                    # Evaluate the SentenceClassifier
                    fake_sentences = torch.argmax(all_outs.detach(), dim=2)
                    classifier_output = sentence_classifier(fake_sentences)
                    classifier_test_loss = F.binary_cross_entropy_with_logits(
                        classifier_output, torch.zeros_like(classifier_output)
                    )
                    running_classifier_test_loss.append(classifier_test_loss.item())

                    if jdx == VALIDATE_AMOUNT:
                        break

            
            avg_classifier_test_loss = np.array(running_classifier_test_loss).mean()
            avg_test_loss = np.array(running_test_loss).mean()
            test_losses.append(avg_test_loss)
            avg_loss = np.array(running_loss).mean()
            train_losses.append(avg_loss)
            print("LABEL: ", dataset.logit_to_sentence(item["logits"][0]))
            print("===")
            print("PRED: ", dataset.logit_to_sentence(all_outs[0]))
            print(f"TRAIN LOSS {avg_loss} | EPOCH {epoch}")
            print(f"TEST LOSS {avg_test_loss} | EPOCH {epoch}")
            print(f"TEST CLASSIFIER LOSS {avg_classifier_test_loss} | EPOCH {epoch}")
            print("BACK TO TRAINING:")
            dataset.train()
        if num_steps % SAVE_INTERVAL == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "num_steps": num_steps,
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                },
                os.path.join("Checkpoints", "Checkpoint_gan" + str(num_steps) + ".pkl"),
            )