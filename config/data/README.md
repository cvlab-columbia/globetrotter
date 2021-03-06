## Naming
There are 52 languages. The datasets are split in 52 splits. The first split has captions for all the languages. 
The other 51 splits have captions for one of the languages that is not English. 
And then, English has captions for all the 52 splits. There are four cases of dataset we may want. 
1. _other = ""_. The regular in which each split is divided in train/val subsplits, and there are 52 splits one for each 
language. Here the English comes from the first split, in case it is selected as a language to use.
2. _other= "same_split"_. All languages for the same split (corresponding captions). Can also select which languages to 
use. For testing translation. This only has one "test" subsplit for each language.
3. _other= "en_same_as"_. Only one language that was held-out from training, in order to finetune on. If it is not 
English, it is equivalent o using point 1. This point focuses on English. If it is English, then we have to use some 
other split that is not the first (because we will want to test with the first later), so we use the split of another 
held-out language.
4. _other = "only_en"_. train/val/test subsplits where all the 52 splits are in English

The purpose of these .json files is to keep a general and sharable-across-tasks configuration of languages.
- "datasets" to use are indicated. Self-explainatory
- "training" and "testing" language splits are indicated. Note the difference between "train" and "training". "train" 
refers to the subsplit, and "training" refers to the languages to use (which splits to use).
- The languages in the "training" split will be divided into within-language-train and within-language-val according to 
the division generated by "split_listings.py". The models will be validated for early stopping as usual (using the 
samples in all the within-language-val for all "training" languages).
- Files are saved with the name of the configuration. Also, there will be a _traininglang and a _testinglang version for 
each dataset (with "training" and "testing" languages), and information about within-language-train vs 
within-language-val (_trainsubsplit vs _valsubsplit)
- There is a special case, which is all English, in which both "training" and "testing" are \["en"\]. In this case, we 
use all the original data for all the datasets. This data is divided in train/val/test independently of the other 
languages (there are separated listing files). Here the variable args.language_split is not important, as both imply
the same split.
- There is the option of using the tokenizer from another json file. This is, the tokenizer obtained from the
text files that are defined in the other json. This is crucial for testing.
- Similarly, there is the option of using precomputed word2vec features from another json file, where the 
language_split to use (from that json) is also specified

In some .json config files we add languages in "testing" even when they are never used for testing. This is so that the
tokenizer generates the tokens also taking into account those languages.
