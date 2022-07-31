# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets
import json
from huggingface_hub import hf_hub_url


_CITATION = """\
@inproceedings{thrush_and_ross2022winoground,
  author = {Tristan Thrush and Ryan Jiang and Max Bartolo and Amanpreet Singh and Adina Williams and Douwe Kiela and Candace Ross},
  title = {Winoground: Probing vision and language models for visio-linguistic compositionality},
  booktitle = {CVPR},
  year = 2022,
}
"""

_URL = "https://huggingface.co/datasets/facebook/winoground"

_DESCRIPTION = """\
Winoground is a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning. Given two images and two captions, the goal is to match them correctly—but crucially, both captions contain a completely identical set of words/morphemes, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing model performance. In our accompanying paper, we probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. In the paper, we perform an extensive analysis to obtain insights into how future work might try to mitigate these models’ shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field.
"""


class WinogroundConfig(datasets.BuilderConfig):
    """BuilderConfig for Winoground."""

    def __init__(self, **kwargs):
        """BuilderConfig for Winoground.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WinogroundConfig, self).__init__(**kwargs)


class Winoground(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = WinogroundConfig

    BUILDER_CONFIGS = [
        WinogroundConfig(
            name="default",
        ),
    ]

    IMAGE_EXTENSION = ".png"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "image_0": datasets.Image(),
                    "image_1": datasets.Image(),
                    "caption_0": datasets.Value("string"),
                    "caption_1": datasets.Value("string"),
                    "tag": datasets.Value("string"),
                    "secondary_tag": datasets.Value("string"),
                    "num_main_preds": datasets.Value("int32"),
                    "collapsed_tag": datasets.Value("string"),
                }
            ),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        hf_auth_token = dl_manager.download_config.use_auth_token
        if hf_auth_token is None:
            raise ConnectionError(
                "Please set use_auth_token=True or use_auth_token='<TOKEN>' to download this dataset"
            )

        downloaded_files = dl_manager.download_and_extract({
            "examples_jsonl": hf_hub_url("datasets/facebook/winoground", filename="data/examples.jsonl"),
            "images_dir": hf_hub_url("datasets/facebook/winoground", filename="data/images.zip")
        })

        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=downloaded_files)]

    def _generate_examples(self, examples_jsonl, images_dir, no_labels=False):
        """Yields examples."""
        examples = [json.loads(example_json) for example_json in open(examples_jsonl).readlines()]
        for example in examples:
            example["image_0"] = os.path.join(images_dir, "images", example["image_0"] + self.IMAGE_EXTENSION)
            example["image_1"] = os.path.join(images_dir, "images", example["image_1"] + self.IMAGE_EXTENSION)
            id_ = example["id"]
            yield id_, example