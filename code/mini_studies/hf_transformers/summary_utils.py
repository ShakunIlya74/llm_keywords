import json
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

SI_IPUT_DF_DIR = Path("../../scholar_inbox/data/scholar_map/llm_inputs")


def write_dict_to_pkl(output_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)


def read_dict_from_pkl(output_path):
    with open(output_path, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict

def read_dict_from_json(output_path):
    with open(output_path, 'r') as f:
        output_dict = json.load(f)
    return output_dict

def read_dict(output_path):
    if output_path.endswith('.pkl'):
        return read_dict_from_pkl(output_path)
    elif output_path.endswith('.json'):
        return read_dict_from_json(output_path)
    else:
        raise ValueError(f"Unsupported file format: {output_path}")


def load_abstracts(n_papers=None, path="../data/llm_inputs/paper_ids_text_pairs.pkl"):
    # load the paper_ids_text_pairs from the file
    with open(path, 'rb') as f:
        paper_ids_text_pairs = pickle.load(f)
    if n_papers:
        paper_ids_text_pairs = list(paper_ids_text_pairs)[:n_papers]
    return paper_ids_text_pairs

def load_texts(n_papers=None, path=Path(SI_IPUT_DF_DIR, "paper_texts_df.parquet")):
    print(f"Loading texts from {path}")
    # load the parquet df using the pyarrow engine to handle multidimensional values
    df = pd.read_parquet(path)
    if n_papers:
        df = df.head(n_papers)
    # parse, paper_id, title, abstract into tuples
    paper_data_tuples = [(row.paper_id, row.title, row.abstract) for row in df.itertuples(index=False)]
    return paper_data_tuples



def parse_llm_outputs_flexible(outputs: Dict[int, str]) -> Dict[int, Dict[str, Optional[Any]]]:
    """
    Parses LLM outputs to extract specified fields using flexible regex patterns.

    Parameters:
    outputs (Dict[int, str]): Dictionary with IDs as keys and LLM output strings as values.

    Returns:
    Dict[int, Dict[str, Optional[Any]]]: Dictionary mapping each ID to a dictionary of extracted fields.
    """

    # Define flexible regex patterns for each field
    patterns = {
        "field of paper": [
            re.compile(r'field[_\s]*of[_\s]*paper\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Field of Paper"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Field\s*of\s*Paper\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'paper[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Paper Field"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)

        ],
        "subfield": [
            re.compile(r'sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Subfield\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'paper[_\s]*sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Paper Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ],
        "sub subfield": [
            re.compile(r'sub[_\s]*sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Sub Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Sub\s*Subfield\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'paper[_\s]*sub[_\s]*sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Paper Sub Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ],
        "keywords": [
            re.compile(r'keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'"Keywords"\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'Keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'paper[_\s]*keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'"Paper Keywords"\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE)
        ],
        "method name  / shortname": [
            re.compile(r'method[_\s]*name[_\s*/]*shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Method name / Shortname"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Method\s*Name\s*/\s*Shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'paper[_\s]*method[_\s]*name\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Paper Method name"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'paper[_\s]*method[_\s]*name[_\s]*shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ]
    }

    parsed_results = {}

    for id_key, text in outputs.items():
        # Initialize the result dictionary with None
        parsed_fields = {
            "field of paper": None,
            "subfield": None,
            "sub subfield": None,
            "keywords": None,
            "method name  / shortname": None,
        }

        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = pattern.search(text)
                if match:
                    value = match.group(1).strip()
                    value = value.replace('*', '')

                    # Post-process keywords to convert to list if possible
                    if field == "keywords":
                        # Remove surrounding brackets or quotes if present
                        value = value.strip("[]\"'")
                        # Split by comma or semicolon and strip whitespace
                        keywords = re.split(r',|;', value)
                        keywords = [kw.strip().strip('"').strip("'") for kw in keywords if kw.strip()]
                        parsed_fields[field] = keywords if keywords else None
                    else:
                        # Remove trailing semicolon if present
                        value = value.rstrip(';').strip()
                        parsed_fields[field] = value if value else None
                    break  # Stop after the first successful match

        parsed_results[id_key] = parsed_fields

    return parsed_results

def create_dirs_if_not_exist():
    import os
    dirs = ['../data/llm_inputs', '../data/llm_outputs', '../data/llm_outputs/keywords', '../data/llm_outputs/statistics',
            '../data/llm_outputs/parsed_summaries',]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def parse_and_save_llm_outputs(output_path, save_to_path):
    outputs = read_dict_from_pkl(output_path)
    parsed_results = parse_llm_outputs_flexible(outputs)
    write_dict_to_pkl(parsed_results, save_to_path)
    return parsed_results


if __name__ == "__main__":
    load_texts()



