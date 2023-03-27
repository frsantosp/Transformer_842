import os
import torch
import torchtext
from Dataset.translation_dataset import EnglishToGermanDataset
from Transformer.transfomer import TransformerTranslator
import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
from torchmetrics.functional.text.rouge import rouge_score
import torchmetrics

nltk.download('wordnet')
nltk.download('omw-1.4')

batch_size = 128
embed_dim = 64
num_blocks = 2
num_heads = 1  # Must be factor of token size
max_context_length = 1000
seed = 842
torch.manual_seed(seed)

# Replace with the path to your checkpoint file
CHECKPOINT_PATH = "Checkpoints/Checkpoint_gan200.pkl"
CUDA=torch.cuda.is_available()

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EnglishToGermanDataset(CUDA=CUDA)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))


# Load your model
encoder_vocab_size = dataset.english_vocab_len
output_vocab_size = dataset.german_vocab_len
torch.set_default_tensor_type(torch.cuda.FloatTensor if CUDA else torch.FloatTensor)
model = TransformerTranslator(
    embed_dim, num_blocks, num_heads, encoder_vocab_size,output_vocab_size,CUDA=CUDA
).to(device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Evaluation function
def evaluate(model, dataloader):
    references = []
    hypotheses = []

    references_tokens = []
    hypotheses_tokens = []
    
    with torch.no_grad():
        for item in dataloader:
            input_tensor = item["english"].to(device)
            target_tensor = item["german"].to(device)
            
            # Generate translation
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
            fake_sentences = torch.argmax(all_outs.detach(), dim=2)
            
            # Convert tensor to text
            for i in range(item["logits"].shape[0]):
                reference = [dataset.word_index_to_sentence(item["german"][i])]
                hypothesis = dataset.word_index_to_sentence(fake_sentences[i])  # Changed from all_outs[i]

                reference_tokens = [dataset.word_index_to_token_list(item["german"][i])]
                hypothesis_tokens = dataset.word_index_to_token_list(fake_sentences[i])  # Changed from all_outs[i]
            
                references.append(reference)
                hypotheses.append(hypothesis)

                references_tokens.append(reference_tokens)
                hypotheses_tokens.append(hypothesis_tokens)

                # print("Reference: {}".format(reference))
                # print("Hypothesis: {}".format(hypothesis))
    
    return references, hypotheses, references_tokens, hypotheses_tokens

# Evaluation
references, hypotheses, references_tokens, hypotheses_tokens = evaluate(model, dataloader)
print("References: ", references_tokens[0])
print("Hypotheses: ", hypotheses_tokens[0])

# Metrics calculation
bleu_metric = torchmetrics.BLEUScore()
chrf_metric = torchmetrics.CHRFScore()
wer = torchmetrics.WordErrorRate()

bleu_score = 0
meteor_score_val = 0
wer_score = 0
n = len(references_tokens)
for i in range(n):
    bleu_score += sentence_bleu(references_tokens[i], hypotheses_tokens[i])
    meteor_score_val += meteor_score.meteor_score(references_tokens[i], hypotheses_tokens[i])
    wer_score += wer(hypotheses[i], references[i])
    
bleu_score /= n
meteor_score_val /= n
wer_score /= n

print(f"BLEU: {bleu_score}")
print(f"METEOR: {meteor_score_val}")
print(f"WER: {wer_score}")

chrf_score = chrf_metric(hypotheses, references)
rouge_scores = rouge_score(hypotheses, references)
# chrf_score = corpus_chrf(references, hypotheses).score

# Print results
print(f"BLEU: {bleu_score}")
print(f"METEOR: {meteor_score_val}")
print(f"ROUGE: {rouge_scores}")
print(f"WER: {wer_score}")
print(f"chrF: {chrf_score}")