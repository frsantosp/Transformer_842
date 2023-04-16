import torch
import spacy
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.metrics.distance import edit_distance
from rouge import Rouge
from torchtext.data.metrics import bleu_score

import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

def translate_sentence_french(model, sentence, french, english, device, max_length=50):
    # Load german tokenizer
    spacy_fre = spacy.load("fr_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def evaluation_scores(data, model, german, english, device):
    targets = []
    outputs = []

    total_meteor = 0.0
    total_wer = 0.0
    rouge_evaluator = Rouge()
    total_rouge = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                   'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                   'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

        # Compute METEOR score
        total_meteor += single_meteor_score(trg, prediction, preprocess=str)

        # Compute WER score
        wer_score = edit_distance(prediction, trg) / max(len(prediction), len(trg))
        total_wer += wer_score

        # Compute ROUGE score
        rouge_scores = rouge_evaluator.get_scores(" ".join(prediction), " ".join(trg), avg=True)
        for key in rouge_scores:
            total_rouge[key]['f'] += rouge_scores[key]['f']
            total_rouge[key]['p'] += rouge_scores[key]['p']
            total_rouge[key]['r'] += rouge_scores[key]['r']

    bleu = bleu_score(outputs, targets)
    meteor = total_meteor / len(data)
    wer = total_wer / len(data)
    rouge = {key: {k: v / len(data) for k, v in total_rouge[key].items()} for key in total_rouge}

    return bleu, meteor, wer, rouge


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])