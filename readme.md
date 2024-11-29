# Seqeval TorchMetrics

## Metric description

This is implemention of [seqeval](https://github.com/chakki-works/seqeval) in [torchmetrics](https://github.com/Lightning-AI/torchmetrics).

seqeval is a Python framework for sequence labeling evaluation. seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on. 

## How to use
Seqeval produces labelling scores along with its sufficient statistics from a source against one or more references.

It takes one mandatory argument:

`labels`: a list of tags, for example `["LOC", "PER", "ORG"]`.

It can also take several optional arguments:

`suffix` (boolean): `True` if the IOB tag is a suffix (after type) instead of a prefix (before type), `False` otherwise. The default value is `False`, i.e. the IOB tag is a prefix (before type).

`scheme`: the target tagging scheme, which can be one of [`IOB1`, `IOB2`, `IOE1`, `IOE2`, `IOBES`, `BILOU`]. The default value is `None`.

`mode`: whether to count correct entity labels with incorrect I/B tags as true positives or not. If you want to only count exact matches, pass `mode="strict"` and a specific `scheme` value. The default is `None`.

`stage`: prefix for keys in output dict. For example `"test"`. The default is `None`.

```python
>>> from metric.seqeval_metric import Seqeval
>>> metric = Seqeval(labels=["MISC", "PER"])
>>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> results = metric(predictions, references)
>>> results
    {'MISC_precision': tensor(0.), 'PER_precision': tensor(1.), 'MISC_recall': tensor(0.), 
    'PER_recall': tensor(1.), 'MISC_f1': tensor(0.), 'PER_f1': tensor(1.), 
    'MISC_number': tensor(1), 'PER_number': tensor(1), 'overall_precision': tensor(0.5000), 
    'overall_recall': tensor(0.5000), 'overall_f1': tensor(0.5000)}
```

## Output values

This metric returns a dictionary with a summary of scores for overall and per type:

Overall:
    
`precision`: the average (micro) [precision](https://en.wikipedia.org/wiki/Precision_and_recall), on a scale between 0.0 and 1.0.
    
`recall`: the average (micro) [recall](https://en.wikipedia.org/wiki/Precision_and_recall), on a scale between 0.0 and 1.0.

`f1`: the average (micro) [F1 score](https://en.wikipedia.org/wiki/Precision_and_recall), which is the harmonic mean of the precision and recall. It also has a scale of 0.0 to 1.0.

Per type (e.g. `MISC`, `PER`, `LOC`,...):

`precision`: [precision](https://en.wikipedia.org/wiki/Precision_and_recall), on a scale between 0.0 and 1.0.

`recall`: [recall](https://en.wikipedia.org/wiki/Precision_and_recall), on a scale between 0.0 and 1.0.

`f1`: [F1 score](https://en.wikipedia.org/wiki/Precision_and_recall), on a scale between 0.0 and 1.0.

`number`: Number of actual positives.


## Limitations and bias

seqeval supports following IOB formats (short for inside, outside, beginning) : `IOB1`, `IOB2`, `IOE1`, `IOE2`, `IOBES`, `IOBES` (only in strict mode) and `BILOU` (only in strict mode). 

For more information about IOB formats, refer to the [Wikipedia page](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) and the description of the [CoNLL-2000 shared task](https://aclanthology.org/W02-2024).

Metric value is substituted as 0 when encountering zero division. 


## Further References 
- [README for seqeval at GitHub](https://github.com/chakki-works/seqeval)
- [CoNLL-2000 shared task](https://www.clips.uantwerpen.be/conll2002/ner/bin/conlleval.txt)
- [huggingface / evaluate](https://github.com/huggingface/evaluate/tree/main/metrics/seqeval)