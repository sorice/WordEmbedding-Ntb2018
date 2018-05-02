# Word Embedded

_wordembedded_ is collection of notebooks to show how to configure word embedding python libraries successfully.

The collection contains practical examples of how to calculate text similarity based on international published algorithms. A final playfull notebook show how to use all the similarity results in a practical NLP problem known as _Paraphrase Recognition_. 

# Requirements

```
$ pip install sklearn gensim pandas scipy glove-python
```

# Usage

  ```
$ python3 preprocess.py <corpus.csv> <model_type> <text_collection> <output.cv>
  ```

__corpus.csv__: this argument intent to capture a sequence of short texts to calculate similarity. In the actual version only MSRPC has been tested.

__model_type__: this argument defines the model type. The options are: _tfidf, glove, word2vec, paragraph2vec_.

__text_collection__: correspond to one option in the defined tags on config.ini. This argument define a path to a text collection to be used in model generation or loading subprocess.

__output.cv__: after all word-embedding methods initialization and manipulation, and all similarity calculations the script generates a matrix with [ text-pair , measures ], that can be used in a _Sklearn_ classifier for Paraprhase Recognition, Semantic Text Similarity, or any other NLP problem based on text pairs similarity.



# Config

* Get your corpus text collection (E.g. Gutenberg, Wikipedia)
  * Add the path to your text collection to the config.ini
* Get the MSRP corpus in a text format loadable by Pandas package.