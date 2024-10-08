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


SPECIAL_TAGS = {
    "SMILES": {"start": "[START_SMILES]", "end": "[END_SMILES]"},
    # "synonym": {"start": "[SYNONYM]", "end": "[/SYNONYM]"},
    "RELATED": {"start": "[RELATED]", "end": "[/RELATED]"},
    "similarity": {"start": "[SIMILAR]", "end": "[/SIMILAR]", "type": float},
    # "PROPERTY": {"start": "[PROPERTY]", "end": "[/PROPERTY]"},
    "SAS": {"start": "[SAS]", "end": "[/SAS]", "type": float},
    "WEIGHT": {"start": "[WEIGHT]", "end": "[/WEIGHT]", "type": float},
    "TPSA": {"start": "[TPSA]", "end": "[/TPSA]", "type": float},
    "CLOGP": {"start": "[CLOGP]", "end": "[/CLOGP]", "type": float},
    "QED": {"start": "[QED]", "end": "[/QED]", "type": float},
    # "NUMHDONORS": {"start": "[NUMHDONORS]", "end": "[/NUMHDONORS]"},
    # "NUMHACCEPTORS": {"start": "[NUMHACCEPTORS]", "end": "[/NUMHACCEPTORS]"},
    # "NUMHETEROATOMS": {"start": "[NUMHETEROATOMS]", "end": "[/NUMHETEROATOMS]"},
    # "NUMROTATABLEBONDS": {
    #     "start": "[NUMROTATABLEBONDS]",
    #     "end": "[/NUMROTATABLEBONDS]",
    # },
    # "NOCOUNT": {"start": "[NOCOUNT]", "end": "[/NOCOUNT]"},
    # "NHOHCOUNT": {"start": "[NHOHCOUNT]", "end": "[/NHOHCOUNT]"},
    # "RINGCOUNT": {"start": "[RINGCOUNT]", "end": "[/RINGCOUNT]"},
    # "HEAVYATOMCOUNT": {"start": "[HEAVYATOMCOUNT]", "end": "[/HEAVYATOMCOUNT]"},
    # "FRACTIONCSP3": {
    #     "start": "[FRACTIONCSP3]",
    #     "end": "[/FRACTIONCSP3]",
    #     "type": float,
    # },
    # "NUMAROMATICRINGS": {
    #     "start": "[NUMAROMATICRINGS]",
    #     "end": "[/NUMAROMATICRINGS]",
    # },
    # "NUMSATURATEDRINGS": {
    #     "start": "[NUMSATURATEDRINGS]",
    #     "end": "[/NUMSATURATEDRINGS]",
    # },
    # "NUMAROMATICHETEROCYCLES": {
    #     "start": "[NUMAROMATICHETEROCYCLES]",
    #     "end": "[/NUMAROMATICHETEROCYCLES]",
    # },
    # "NUMAROMATICCARBOCYCLES": {
    #     "start": "[NUMAROMATICCARBOCYCLES]",
    #     "end": "[/NUMAROMATICCARBOCYCLES]",
    # },
    # "NUMSATURATEDHETEROCYCLES": {
    #     "start": "[NUMSATURATEDHETEROCYCLES]",
    #     "end": "[/NUMSATURATEDHETEROCYCLES]",
    # },
    # "NUMSATURATEDCARBOCYCLES": {
    #     "start": "[NUMSATURATEDCARBOCYCLES]",
    #     "end": "[/NUMSATURATEDCARBOCYCLES]",
    # },
    # "NUMALIPHATICRINGS": {
    #     "start": "[NUMALIPHATICRINGS]",
    #     "end": "[/NUMALIPHATICRINGS]",
    # },
    # "NUMALIPHATICHETEROCYCLES": {
    #     "start": "[NUMALIPHATICHETEROCYCLES]",
    #     "end": "[/NUMALIPHATICHETEROCYCLES]",
    # },
    # "NUMALIPHATICCARBOCYCLES": {
    #     "start": "[NUMALIPHATICCARBOCYCLES]",
    #     "end": "[/NUMALIPHATICCARBOCYCLES]",
    # },
    # "IUPAC": {"start": "[IUPAC]", "end": "[/IUPAC]"},
    # "VAR_NAME": {"start": "[VAR_NAME]", "end": "[/VAR_NAME]"},
    # "VAR_DESC": {"start": "[VAR_DESC]", "end": "[/VAR_DESC]"},
    # "VAR_VAL": {"start": "[VAR_VAL]", "end": "[/VAR_VAL]"},
    # "ASSAY_NAME": {"start": "[ASSAY_NAME]", "end": "[/ASSAY_NAME]"},
    # "ASSAY_DESC": {"start": "[ASSAY_DESC]", "end": "[/ASSAY_DESC]"},
    "formula": {"start": "[FORMULA]", "end": "[/FORMULA]"},
}


@cache
def get_special_tags(molecular_repr):
    with open(os.path.expanduser("~/YNNtitan/torchtitan/tokenizers/special_tokens.toml"), "rb") as f:
        special_tokens = tomllib.load(f)
    
    tags_to_include = [molecular_repr, "related", "SAS", "WEIGHT", "TPSA", "CLOGP", "QED", "RINGCOUNT", "formula"]
    return {prop: special_tokens[prop] for prop in tags_to_include}


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
        special_tags = get_special_tags(representation_type)
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