from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from rich.logging import RichHandler
import logging
from collections import Counter
from torch import cuda
import numpy as np
import pdb
from transformers import DataCollatorForTokenClassification
import evaluate
import config

# Rich Handler for colorized logging, you can safely remove it
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

# Please safely remore the following path and add path where you want to store the data from hugginface and models
path = "/mount/studenten/arbeitsdaten-studenten1/razaai/cache"

# label mappings
label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}
id2label = {v: k for k, v in label2id.items()}

# check for GPU
device = 'cuda' if cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=path)
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=31, id2label=id2label, label2id=label2id, cache_dir=path
)

# Define DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# load metric
seqeval = evaluate.load("seqeval") 

# define hyperparameters
MAX_LEN = 128
MAX_GRAD_NORM = 10

def load_data(dataset_name=None):
    # A function that loads the data and returns its objects. 
    # Please add path in cache_dir where you want to store the dataset
    try:
        return load_dataset("Babelscape/multinerd", cache_dir=path)
    except Exception as e:
        print("failed to load the data: %s" % (str(e)))

def preprcossing(data):
    # A function that filters the data by remove non-English instances
    try:
        return data.filter(lambda example: example['lang'] == "en")
    except Exception as e:
        print("failed to preprocess the data: %s" % (str(e)))

def data_statistics(data):
    # A funtion that provides statistics of NER tags e.g., representation of each class
    ner_tags_lst = []
    for lst in data['ner_tags']:
        lst = sorted(lst, reverse=True)
        index = 0
        while lst[index] != 0:
            tag = next((key for key, label in label2id.items() if label == lst[index]), None)
            index+=1
            ner_tags_lst.append(tag)
    logger.info(f"Representation of each tag in the dataset: {Counter(ner_tags_lst)}")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
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

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train(tokenized_train, tokenized_val):
    # The function implements Trainer class from Hugginface for finetuning the model
    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # push the model o huggingface
    trainer.push_to_hub()

if __name__ == "__main__":
    data = load_data()
    logger.info(f"Train dataset size: {len(data['train'])}")
    logger.info(f"Validation dataset size: {len(data['validation'])}")
    logger.info(f"Test dataset size: {len(data['test'])}")

    # get the list of all labels
    # label_lst = list(label2id.keys())
    # labels = [label_lst[i] for i in data['train']["ner_tags"]]

    # preprocessing
    data_train = preprcossing(data['train'])
    logger.info(f"Train dataset size after preprocessing: {data_train}")

    data_val = preprcossing(data['validation'])
    logger.info(f"Validation dataset size after preprocessing: {data_val}")

    data_test = preprcossing(data['test'])
    logger.info(f"Test dataset size after preprocessing: {data_test}")

    # get to know data 
    data_statistics(data_train)
    logger.info(f"This is how a single training example looks like: {data_train[0]}")

    # remove lang column as we dont need it anymore 
    data_train =data_train.remove_columns("lang")
    data_val =data_val.remove_columns("lang")
    data_test =data_test.remove_columns("lang")

    logger.info(f"Training Dataset Shape: {data_train.shape}")
    logger.info(f"Validation Dataset Shape: {data_val.shape}")
    logger.info(f"Test Dataset Shape: {data_test.shape}")

    tokenized_train = data_train.map(tokenize_and_align_labels, batched=True)
    tokenized_val = data_val.map(tokenize_and_align_labels, batched=True)
    tokenized_test = data_test.map(tokenize_and_align_labels, batched=True)

    finetuning script
    train(tokenized_train, tokenized_val)
