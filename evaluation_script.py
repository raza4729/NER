import evaluate
from rich.logging import RichHandler
import logging
import config
from datasets import load_dataset
from transformers import pipeline
from torch import cuda
import argparse

# load metric
seqeval = evaluate.load("seqeval") 

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


# label mappings
label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

# in order to convert leave-out labels to zero following list will be used as reference
blocked_labels = {"B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

id2label = {v: k for k, v in label2id.items()}

device = 'cuda' if cuda.is_available() else 'cpu'

# get the list of all labels
label_list = list(label2id.keys())

def turn_off_labels(labels=None):
    # A function that converts leave-out labels to Zero
    revised_labels = []
    for label in labels:
        if label in blocked_labels.values():
            revised_labels.append(0)
        else:
            revised_labels.append(label)
    return revised_labels

def load_data(dataset_name=None):
    # A function that loads the data and returns its objects. 
    # Please add path in cache_dir where you want to store the dataset
    try:
        return load_dataset("Babelscape/multinerd", cache_dir=config.path)
    except Exception as e:
        print("failed to load the data: %s" % (str(e)))

def preprcossing(data):
    # A function that filters the data by remove non-English instances
    try:
        return data.filter(lambda example: example['lang'] == "en")
    except Exception as e:
        print("failed to preprocess the data: %s" % (str(e)))

def predict(test_set, model_name):

    logger.info(f"Test set: {test_set}")
    data_test = preprcossing(test_set)
    
    data_test =data_test.remove_columns("lang")
    logger.info(f"Test dataset size after preprocessing: {data_test}")
    
    lst_predictions = lst_labels = []
    nlp = pipeline("ner", model=model_name)

    for instance in data_test:
        text = ' '.join(instance['tokens'])
        labels = instance['ner_tags']
        
        # to switch unneccesary labels to O
        # labels = turn_off_labels(labels)
        
        true_labels = [id2label[x] for x in labels]
        output = nlp(text)

        predictions = []
        
        for token in instance['tokens']:
            switch = 0
            for d in output:
                if token.lower() == d['word']:
                    predictions.append(d['entity'])
                    switch = 1
            if switch == 0:
                predictions.append('O')

        lst_predictions.append(predictions)
        lst_labels.append(true_labels)
        

    results = seqeval.compute(predictions=lst_predictions, references=lst_labels, zero_division=0)
    logger.info(f"Evaluation: {results}")
   
    f = open("evaluation_on_test.txt", "a")
    f.write(str(model_name))
    f.write("\n")
    f.write("precision: " + str(results["overall_precision"]))
    f.write("\n")
    f.write("recall: " + str(results["overall_recall"]))
    f.write("\n")
    f.write("f1: " + str(results["overall_f1"]))
    f.write("\n")
    f.write("accuracy:" + str(results["overall_accuracy"]))
    f.write("\n")

def parse_args():
    # takes command-line argument for leave-out labels whether to remove them or not
    parser = argparse.ArgumentParser(description="a script which takes arguments as model name")
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    return args.model_name

if __name__ == "__main__":

    data = load_data()
    model_name = parse_args()
    predict(data['test'], model_name)