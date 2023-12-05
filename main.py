from datasets import load_dataset
from rich.logging import RichHandler


# Rich Handler for colorized logging, you can safely remove it
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def load_data():
    # A function that loads the data and returns its objects
    try:
        return load_dataset("Babelscape/multinerd", cache_dir="/mount/studenten/arbeitsdaten-studenten1/razaai/cache")
    except Exception as e:
        print("failed to load the data: %s" % (str(e)))

def preprcossing(*args):
    # A function that filters the data by remove non-English instances
    pass

def load_model(*args):
    # A funtion that loads the model from hugginface
    pass

if __name__ == "__main__":
    data = load_data()
    print(data)