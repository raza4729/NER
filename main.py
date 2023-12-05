from datasets import load_dataset
from rich.logging import RichHandler
import logging

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

def load_data():
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

def load_model(model_name):
    # A funtion that loads the model from hugginface
    pass

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