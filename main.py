from datasets import load_dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from rich.logging import RichHandler
import logging
from collections import Counter
from torch import cuda


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
mappings = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

# check for GPU
device = 'cuda' if cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")

# load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=path)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', cache_dir=path)

# define hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
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
            tag = next((key for key, label in mappings.items() if label == lst[index]), None)
            index+=1
            ner_tags_lst.append(tag)
    logger.info(f"Representation of each tag in the dataset: {Counter(ner_tags_lst)}")

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def load_tokenizer():
    # A funtion that loads the model & tokenizer from hugginface
    try:
        pass
    except Exception as e:
        print("failed to preprocess the data: %s" % (str(e)))

if __name__ == "__main__":
    data = load_data()
    logger.info(f"Train dataset size: {len(data['train'])}")
    logger.info(f"Validation dataset size: {len(data['validation'])}")
    logger.info(f"Test dataset size: {len(data['test'])}")

    # preprocessing
    data_train = preprcossing(data['train'])
    logger.info(f"Train dataset size after preprocessing: {data_train}")

    data_val = preprcossing(data['validation'])
    logger.info(f"Validation dataset size after preprocessing: {data_val}")

    data_test = preprcossing(data['test'])
    logger.info(f"Test dataset size after preprocessing: {data_test}")

    # get to know data 
    data_statistics(data_train)
    logger.info(f"This is how a single training example looks like: {data_train[0] }")
    

