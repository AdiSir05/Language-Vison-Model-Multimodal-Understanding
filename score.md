## Evaluation Metrics
We evaluate the generated captions using the following metrics:

### BLEU-2 and BLEU-3
##### Definition: 
BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text by measuring the overlap between the generated text and one or more reference texts. BLEU-2 and BLEU-3 consider up to 2-grams and 3-grams, respectively.
##### Calculation: 
The BLEU score is calculated based on the precision of n-grams between the generated and reference captions, with a brevity penalty for short sentences.
##### Reference: 
Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation," ACL 2002.

### METEOR
##### Definition: 
METEOR (Metric for Evaluation of Translation with Explicit ORdering) evaluates machine translation output by aligning words and calculating precision and recall, with a higher weight on recall.
##### Calculation: 
METEOR computes a score based on unigram matches between the generated and reference captions, considering synonyms and stemming, and includes a fragmentation penalty.
##### Reference: 
Banerjee and Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments," ACL Workshop 2005.

### ROUGE
##### Definition: 
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the quality of a summary by comparing it to reference summaries.
##### Calculation: 
ROUGE-N measures n-gram recall between the generated and reference captions. ROUGE-L measures the longest common subsequence.
##### Reference: 
Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL Workshop 2004.

## Using the Scripts
Follow the function definition, and use them as helper functions that would be imported to your code
example: "

## Example Usage 
generated_captions would be a list of generated captions (str):
```
[ "a young boy standing on a beach with a surfboard", "a woman in a dress standing on a street", ..... ]
```

reference_captions would be a list of list of reference captions (str, totally 5 reference caption per generated caption)
```
[ ['A boy in his blue swim shorts at the beach .',
'A boy smiles for the camera at a beach .',
'A young boy in swimming trunks is walking with his arms outstretched on the beach .',
'Children playing on the beach .',
'The boy is playing on the shore of an ocean .'] ,

['A blond woman in a blue shirt appears to wait for a ride .', 
'A blond woman is on the street hailing a taxi .',
'A woman is signaling is to traffic , as seen from behind .',
'A woman with blonde hair wearing a blue tube top is waving on the side of the street .', 
'The woman in the blue dress is holding out her arm at oncoming traffic .'] ,

...]
```

After passing these two arguments into compute_evaluation_metrics(generated_captions, reference_captions), you would get a dictionary of performance metrics, like
```
{       "meteor": 0.33,
        "bleu2": 0.5,
        "bleu3": 0.3,
        "rouge1": 0.4,
        "rouge2": 0.3,
        "rougeL": 0.3      }
```




