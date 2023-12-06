# Named Entity Recognition

The repository offers implementation of named entity recognition task using Huggingface and Pytorch. We used [Multinerd](https://huggingface.co/datasets/Babelscape/multinerd?row=0) dataset from Hugginface for this task. We finetuned the off-the-shelf model (Distilbert) in two ways; one using all the outcome labels and one with leaving-out outcome labels which are considered part of general entities (in other words hyponyms).  The finetuned models are available on [Huggingface](https://huggingface.co/imrazaa) to be used as off-the-shelf models without finetuning. 

## Reuseability

The script in the main.py is used to finetune the models and could be resued in a following manner. 

1. ```Create virtual environment```

2. ```Install requirements.txt by running -> pip install -r requirements.txt```

3. ```python main.py --leaveOut 1/0```

The `--leaveOut` arguemnt is intended to implement leave-out functionality for leaving out certain outcome labels which might lead to data imbalance or are considerend trivial depending on the requirments.