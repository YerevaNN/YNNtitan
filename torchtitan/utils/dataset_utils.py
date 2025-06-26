# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/YerevaNN/ChemLactica/blob/main/chemlactica/utils/dataset_utils.py
# All rights reserved

# import orjson
import json

from torchtitan.logging import logger

from .text_format_utils import (
    delete_empty_tags,
    generate_formatted_conformer_string,
    generate_formatted_string,
    sample_dict_to_formatted_string,
)


def load_jsonl_line(jsonl_line):
    try:
        _maybe_compound_dict = orjson.loads(jsonl_line)
        if isinstance(_maybe_compound_dict, dict):
            return _maybe_compound_dict
        else:
            return orjson.loads(_maybe_compound_dict)
    except json.JSONDecodeError as e:  # noqa
        raise ValueError(f"Error decoding JSON: {e}")  # noqa


def chemlactica_style_data_processing(sample_json, rng, representation_type):
    try:
        sample_json = json.loads(sample_json)
        compound = delete_empty_tags(sample_json)
        return generate_formatted_string(compound, rng, representation_type)
    except Exception as e:
        logger.info(e)
    return ""


def conformer_data_processing(sample_json, rng, representation_type):
    try:
        compound = json.loads(sample_json)
        sample_json = generate_formatted_conformer_string(
            compound, rng, representation_type
        )
    except Exception as e:
        print(e)
        sample_json = ""
    return sample_json


def sft_formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["smiles"])):
        text = (
            f"<bos>[START_SMILES]{example['smiles'][i]}[END_SMILES]"
            "[PROPERTY]activity {round(example['activity'][i], 2)}[/PROPERTY]"
        )
        output_texts.append(text)
    return output_texts


def pubchem_data_processing(sample_json, rng, representation_type):
    try:
        sample_dict = json.loads(sample_json)
        sample_dict = delete_empty_tags(sample_dict)
        return sample_dict_to_formatted_string(sample_dict, rng)
    except Exception as e:
        logger.info(e)
    return ""
