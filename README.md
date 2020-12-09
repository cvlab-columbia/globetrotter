# Globetrotter

Code from the paper [Learning to Learn Words from Visual Scenes](https://arxiv.org/abs/2012.04631).

Website of the project in [globetrotter.cs.columbia.edu](https://globetrotter.cs.columbia.edu).

If you use the code or the dataset, please consider citing the paper as:

```
@article{suris2019learning,
  title={Globetrotter: Unsupervised Multilingual Translation from Visual Alignment},
  author={Sur\'is, D\'idac and Epstein, Dave and Vondrick, Carl},
  journal={arXiv preprint arXiv:2012.04631},
  year={2020}
}
```

An example of command line execution to train the mdoel can be found in `scripts/train_globetrotter.sh`. To reproduce 
the numbers from the paper, please use the released pretrained models, and the `scripts/test/*.sh` scripts. In order to 
run those scripts, extract features by running `scripts/extract_features/*.sh` first. Modify the parameters in the bash
files with the corresponding paths (dataset, extracted features)

Run python `main.py --help` for information on arguments.

Be sure to have the external libraries in `requirements.txt` installed.

## Data

We collected the Globetrotter dataset for this project. It contains captions in 52 different languages for images from 
three different captioning datasets: [MSCOCO](https://cocodataset.org), [Flickr30k](http://bryanplummer.com/Flickr30kEntities/) and 
[Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions). In order to train new models, you will
need to download the images from the corresponding links. 

Our collected captions can be downloaded from [this link (dataset.tar.gz)](https://globetrotter.cs.columbia.edu/dataset.tar.gz). The provided file already contains
the folder structure that is required to execute our code, that follows the folder structure of the original datasets. That file also
contains the human test translations. The folder `translated_independent` contains sentences that describe different images
in each language. `translated_alllangs` contains translations that describe the same image for all languages (for testing purposes).
 
Other dataset information necessary to run our models (splits, tokenizer and word2vec information) can be found in 
[this link (dataset_info.tar.gz)](https://globetrotter.cs.columbia.edu/dataset_info.tar.gz)

As a reminder, you can extract the content from a `.tar.gz` file by using `tar -xzvf archive.tar.gz`.

The root dataset directory can be given by using the argument `--dataset_path`. Use `--dataset_info_path` to indicate
 the path to the dataset information files. In order to use the code without images,
you can use the flag `not_use_images`.

## Pretrained models

The pretrained models reported in our paper can be found in [this link (checkpoints.tar.gz)](https://globetrotter.cs.columbia.edu/checkpoints.tar.gz):

Each folder (one for each model) contains a `.pth` file with the checkpoint, as well as a `.json` file with the configuration.

To resume training or to test from one of these pretrained models, set the `--resume` flag to `True`. Extract the models 
under the `/path/to/your/checkpoints` directory you introduce in the `--checkpoint_dir` argument. Refer to the specific model using the `--resume_name` argument.

In case there is any doubt or problem, feel free to send us an email.

