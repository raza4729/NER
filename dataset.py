from torch.utils.data import Dataset
import torch
import config 
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
import pdb

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.path)
model = AutoModelForTokenClassification.from_pretrained(
    config.model, num_labels=len(config.label2id), id2label=config.id2label, label2id=config.label2id, cache_dir=config.path
)


class dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, leaveOut=0):
        self.len = len(data) # length of data
        self.data = data # dataset objecct
        self.tokenizer = tokenizer # tokenizer
        self.max_len = max_len # max length of sequence
        self.leaveOut = leaveOut # whether to convert selected ner_tags to zero or not
            
    def __getitem__(self, index):
        """ A function to tokenize the data in order to be accepted by model
        The code is taken from Hugginface and modified """
        
        tokens = self.data["tokens"][index]
        tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True)
        labels = []
        
        for i, label in enumerate(self.data["ner_tags"]):
           
            if self.leaveOut:    
                label = self.set_labels_to_zero(label)  

            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            pdb.set_trace()
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
      
    def set_labels_to_zero(self, labels):
        # A function that converts leave-out labels to Zero
        revised_labels = []
        for label in labels:
            if label in config.blocked_labels.values():
                revised_labels.append(0)
            else:
                revised_labels.append(label)
        return revised_labels

    def __len__(self):
        return self.len
    
data = load_dataset("Babelscape/multinerd", cache_dir=config.path)
training_set = dataset(data['train'], tokenizer, 128, 0)
print(training_set[0])
# print(training_set[1])