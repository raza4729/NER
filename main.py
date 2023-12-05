from datasets import load_dataset
from transformers import AutoTokenizer
from rich.logging import RichHandler
import logging
from collections import Counter

# Rich Handler for colorized logging, you can safely remove it
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

# add path where you want to store the data from hugginface and models
path = "/mount/studenten/arbeitsdaten-studenten1/razaai/cache"

# label mappings
mappings = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

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
    

def load_model(model_name):
    # A funtion that loads the model & tokenizer from hugginface
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = ""
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

    # load model 
