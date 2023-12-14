# Named Entity Recognition

The repository offers implementation of named entity recognition task using Huggingface and Pytorch. We used [Multinerd](https://huggingface.co/datasets/Babelscape/multinerd?row=0) dataset from Hugginface for this task. We finetuned the off-the-shelf model (Distilbert) in two ways; one using all the outcome labels and one with leaving-out outcome labels which are considered part of general entities (in other words hyponyms).  The finetuned models are available on [Huggingface](https://huggingface.co/imrazaa) to be used as off-the-shelf models without finetuning. 

We used DistilBert for the experiments, please feel free to use any model by changing the name in `config.py` file. 
## Reuseability

The script in the main.py is used to finetune the models and could be resued in a following manner. 

1. ```Create virtual environment```

2. ```Install requirements.txt by running -> pip install -r requirements.txt```

3. ```python main.py --leaveOut 0```

The `--leaveOut` arguemnt is intended to implement leave-out functionality for skiping certain outcome labels which might lead to data imbalance or are considerend trivial depending on the requirments. Set it to `1` to leave them and `0` to finetune model with all labels.

## Remarks 

The model that models all of the labels when evaluated solely using accuracy could be misleading due to class imbalance as in this case, the entities that are hyponyms would always be sparse as compared to general entities which is depicted by the accuracy of [Model A](https://huggingface.co/imrazaa/named-entity-recognition-distilbert-A) if we compare it with F1-score. The F1 score on both validation (dev) dataset and test dataset are reported below. 

Additionaly, the model might improve as we experiment with different hyperparamethers as in this case all of them were kept constant across all experiments. Adding more to that, the model only accounts for token level entities for instance in case of `Aggregate` tokens such as `Paris` but not for `New York University` where these models might struggle, however this something we are hypothesizing as there was no qualitiative analysis report yet.

| Model | Dev Data | Test Data |
| ----- | -------- | --------- |
| A     |  0.89    |           |
| B     |  0.94    |           |


To test the finetuned model on test data (Multinerd) please run the following commands in cmd after activating virtual env.. and installing requirements. 

```python evaluation_script.py --model_name named-entity-recognition-distilbert-A```

or

```python evaluation_script.py --model_name named-entity-recognition-distilbert-B```


### Citation 
### Bibtex
```
@software{Ali_Raza,
    author = {Raza, Ali},
    license = {BSD-2-Clause license},
    title = {Named Entity Recognition using Multinerd},
    url = {https://github.com/raza4729/NER}
}
```