# Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality
## Dataset Description
Winoground is a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning. Given two images and two captions, the goal is to match them correctly—but crucially, both captions contain a completely identical set of words/morphemes, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing model performance. In our accompanying paper, we probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. In the paper, we perform an extensive analysis to obtain insights into how future work might try to mitigate these models’ shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field.  The dataset is available at https://huggingface.co/datasets/facebook/winoground.


## Winoground Explorer
Slide across the slider to see various examples from WinoGround: https://huggingface.co/spaces/CVPR/winoground-explorer.

## Data
The dataset has 1600 image-text pairs in total, with 800 correct and 800 incorrect pairings. These comprise
400 examples, with 800 unique captions and images. The captions and tags are located in `data/examples.jsonl` and the images are located in `data/images.zip`. You can load the data as follows:
```
from datasets import load_dataset
examples = load_dataset('facebook/winoground', use_auth_token=<YOUR USER ACCESS TOKEN>)
```
You can get `<YOUR USER ACCESS TOKEN>` by following these steps:
1) log into your Hugging Face account
2) click on your profile picture
3) click "Settings"
4) click "Access Tokens"
5) generate an access token

## Model Notebooks
Model notebooks include necessary code to run Winoground evaluation with different versions of ViLT, CLIP, FLAVA and BLIP. ViLT, CLIP and FLAVA are also included in the Winoground paper. More versions of ViLT and CLIP are tested here. BLIP evaluation is not included in the Winoground paper. All versions of BLIP are tested here. These includes models of different sizes. Some models are pre-trained checkpoints and others have been finetuned on COCO and Flickr30k for retrieval.

## Model Predictions and Statistics
The image-caption model scores from our paper are saved in `statistics/model_scores`. To compute many of the tables and graphs from our paper, run the following commands:

```
pip install -r statistics/requirements.txt
python statistics/compute_statistics.py
```

The new results are saved in `results`. To compute similar tables from the paper run `winoground_statistics.ipynb`.

## Citation Information
Tristan Thrush and Candace Ross contributed equally.
```
@inproceedings{thrush_and_ross2022winoground,
  author = {Tristan Thrush and Ryan Jiang and Max Bartolo and Amanpreet Singh and Adina Williams and Douwe Kiela and Candace Ross},
  title = {Winoground: Probing vision and language models for visio-linguistic compositionality},
  booktitle = {CVPR},
  year = 2022,
}
```