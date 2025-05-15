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



import re
import json
from typing import Dict, Any, Optional

def parse_llm_outputs_flexible(outputs: Dict[int, str]) -> Dict[int, Dict[str, Optional[Any]]]:
    """
    Parses LLM outputs to extract specified fields.
    First attempts to parse a JSON object; if that fails, uses regex patterns.

    Parameters:
    outputs: mapping from an ID to the raw LLM output string.

    Returns:
    mapping from each ID to a dict with keys:
      - "field of paper"
      - "subfield"
      - "sub subfield"
      - "keywords"
      - "method name  / shortname"
    Values are strings (or list of strings for keywords), or None if not found.
    """

    # --- compile your regex patterns once ---
    patterns = {
        "field of paper": [
            re.compile(r'field[_\s]*of[_\s]*paper\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            # …all your other field-of-paper patterns…
        ],
        "subfield": [
            re.compile(r'sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            # …etc…
        ],
        "sub subfield": [
            re.compile(r'sub[_\s]*sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            # …etc…
        ],
        "keywords": [
            re.compile(r'keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            # …etc…
        ],
        "method name  / shortname": [
            re.compile(r'method[_\s]*name[_\s*/]*shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            # …etc…
        ],
    }

    # mapping from normalized JSON keys → your field names
    json_key_map = {
        "field_of_paper":          "field of paper",
        "subfield":                "subfield",
        "sub_subfield":            "sub subfield",
        "keywords":                "keywords",
        "method_name_shortname":   "method name  / shortname",
        "method_name_short_name":  "method name  / shortname",  # in case of variant
    }

    def _init_fields():
        return {k: None for k in patterns.keys()}

    parsed_results: Dict[int, Dict[str, Optional[Any]]] = {}

    for id_key, text in outputs.items():
        parsed_fields = _init_fields()

        # --- 1) Try JSON extraction ---
        # capture inside ```json ... ``` if present
        m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        json_str = m.group(1) if m else None

        # otherwise grab first {...} block
        if not json_str:
            m2 = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\})', text, re.DOTALL)
            if m2:
                json_str = m2.group(1)

        if json_str:
            try:
                data = json.loads(json_str)
                # normalize keys and assign
                for raw_k, raw_v in data.items():
                    norm_k = raw_k.strip().lower().replace(" ", "_")
                    if norm_k in json_key_map:
                        field_name = json_key_map[norm_k]
                        val = raw_v
                        if field_name == "keywords":
                            # ensure a list
                            if isinstance(val, str):
                                # split comma/semicolon if user provided a single string
                                val = [v.strip() for v in re.split(r'[;,]', val) if v.strip()]
                            # else assume it’s already a list
                        else:
                            # coerce to plain string
                            val = str(val).strip()
                        parsed_fields[field_name] = val or None
                # if at least one field was populated, skip regex step
                if any(parsed_fields.values()):
                    parsed_results[id_key] = parsed_fields
                    continue
            except json.JSONDecodeError:
                # malformed JSON: fall back to regex
                pass

        # --- 2) Fallback to regex matching ---
        for field, field_patterns in patterns.items():
            for pat in field_patterns:
                m = pat.search(text)
                if not m:
                    continue
                value = m.group(1).strip().rstrip(';')
                if field == "keywords":
                    # strip surrounding brackets/quotes and split
                    value = value.strip('[]"\' ')
                    items = [kw.strip().strip('"\'') for kw in re.split(r'[;,]', value) if kw.strip()]
                    parsed_fields[field] = items or None
                else:
                    parsed_fields[field] = value or None
                break

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



