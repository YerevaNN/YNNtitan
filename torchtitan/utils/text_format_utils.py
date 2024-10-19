# Adapted from https://github.com/YerevaNN/ChemLactica/blob/main/chemlactica/utils/text_format_utils.py
# All rights reserved
from torchtitan.logging import logger
from functools import cache
import safe

import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@cache
def read_special_tags():
    with open(os.path.expanduser("~/YNNtitan/torchtitan/tokenizers/special_tokens.toml"), "rb") as f:
        special_tokens = tomllib.load(f)

    return special_tokens


@cache
def get_tags_split(molecular_repr):
    included_properties = {molecular_repr, "related", "SAS", "WEIGHT", "TPSA", "CLOGP", "QED", "RINGCOUNT", "formula"}
    all_properties = set(read_special_tags().keys())
    return list(included_properties), list(all_properties.difference(included_properties))


def sample_special_tags(molecular_repr, rng, sample_p=0.1):
    included_properties, sampled_properties = get_tags_split(molecular_repr)

    # sample the properties
    do_sample = rng.random(len(sampled_properties)) < sample_p
    sampled_properties = [p for i, p in enumerate(sampled_properties) if do_sample[i]]

    return {key: read_special_tags()[key] for key in included_properties + sampled_properties}


def delete_empty_tags(compound_json):
    for k, v in list(compound_json.items()):
        if v == [] or v == "":
            del compound_json[k]
    return compound_json


def convert_representation(smiles, representation_type):
    try:
        return {
            "SMILES": lambda x: x,
            "SAFE": safe.encode
        }[representation_type](smiles)
    except Exception as e:
        # logger.info(f"{e}. Could not encode molecule {smiles} with representation {representation_type}")
        return smiles


def generate_formatted_string(compound_json, rng, representation_type):
    key_value_pairs = []

    # add smiles in the beginning 50% of the time
    key = "SMILES"
    value = compound_json.get(key, "")

    if rng.integers(2) == 0:
        if value:
            key_value_pairs.append(format_key_value(key, value, rng, representation_type))
            del compound_json[key]

    keys = list(compound_json.keys())
    rng.shuffle(keys)

    for key in keys:
        key_value_pairs.append(format_key_value(key, compound_json[key], rng, representation_type))
    compound_formatted_string = (
        "".join(key_value_pairs)
    )
    return compound_formatted_string


def format_key_value(key, value, rng, representation_type):
    if key == "SMILES":
        key = representation_type

    formatted_string = ""
    try:
        special_tags = sample_special_tags(representation_type, rng)
        if special_tags.get(key):
            start_tag = special_tags[key]['start']
            end_tag = special_tags[key]['end']
            if key == representation_type:
                formatted_string = f"{start_tag}{convert_representation(value, representation_type)}{end_tag}"
            elif key == "related":
                if len(value) > 10:
                    value = rng.choice(value, size=10, replace=False, shuffle=False)
                for pair in value:
                    rounded_sim = "{:.2f}".format(float(pair["similarity"]))
                    mol_repr = convert_representation(pair["SMILES"], representation_type)
                    formatted_string += f"{start_tag}{mol_repr} {rounded_sim}{end_tag}"  # noqa
            elif key == "experimental":
                for pair in value:
                    formatted_string += f"{start_tag}{pair['PROPERTY_NAME']} {pair['PROPERTY_VALUE']}{end_tag}"  # noqa
            elif key == "synonyms":
                for val in value:
                    formatted_string += f"{start_tag}{val['name']}{end_tag}"  # noqa
            else:
                if special_tags[key].get("type") == "float":
                    value = "{:.2f}".format(float(value))
                    assert len(value.split(".")[-1]) == 2
                formatted_string = f"{start_tag}{value}{end_tag}"
    except Exception as e:
        logger.info(e)

    return formatted_string