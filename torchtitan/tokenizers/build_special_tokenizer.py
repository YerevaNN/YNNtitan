import argparse
from transformers import AutoTokenizer

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--special_tokens_toml_path", required=False, type=str, default=None)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.special_tokens_toml_path:
        with open(args.special_tokens_toml_path, "rb") as f:
            special_tokens = tomllib.load(f)

        for value in special_tokens.values():
            tokenizer.add_tokens([value["start"], value["end"]], special_tokens=True)

    tokenizer.save_pretrained(args.save_path)
