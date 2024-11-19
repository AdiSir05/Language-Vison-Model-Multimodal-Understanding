## Evaluation Metrics
We evaluate the generated captions using the following metrics:

### BLEU-2 and BLEU-3
Definition: BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text by measuring the overlap between the generated text and one or more reference texts. BLEU-2 and BLEU-3 consider up to 2-grams and 3-grams, respectively.
Calculation: The BLEU score is calculated based on the precision of n-grams between the generated and reference captions, with a brevity penalty for short sentences.
Reference: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation," ACL 2002.

### METEOR
Definition: METEOR (Metric for Evaluation of Translation with Explicit ORdering) evaluates machine translation output by aligning words and calculating precision and recall, with a higher weight on recall.
Calculation: METEOR computes a score based on unigram matches between the generated and reference captions, considering synonyms and stemming, and includes a fragmentation penalty.
Reference: Banerjee and Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments," ACL Workshop 2005.

### ROUGE
Definition: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the quality of a summary by comparing it to reference summaries.
Calculation: ROUGE-N measures n-gram recall between the generated and reference captions. ROUGE-L measures the longest common subsequence.
Reference: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL Workshop 2004.