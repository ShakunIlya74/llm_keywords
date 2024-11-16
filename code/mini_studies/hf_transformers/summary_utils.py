import pickle
import re
from typing import Dict, Optional, Any


def write_dict_to_pkl(output_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)


def read_dict_from_pkl(output_path):
    with open(output_path, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict


def load_abstracts(n_papers=None):
    # load the paper_ids_text_pairs from the file
    with open("../data/llm_inputs/paper_ids_text_pairs.pkl", 'rb') as f:
        paper_ids_text_pairs = pickle.load(f)
    if n_papers:
        paper_ids_text_pairs = paper_ids_text_pairs[:n_papers]
    return paper_ids_text_pairs


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
            re.compile(r'Field\s*of\s*Paper\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ],
        "subfield": [
            re.compile(r'sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Subfield\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ],
        "sub subfield": [
            re.compile(r'sub[_\s]*sub[_\s]*field\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Sub Subfield"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Sub\s*Subfield\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        ],
        "keywords": [
            re.compile(r'keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'"Keywords"\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE),
            re.compile(r'Keywords\s*[:=]\s*(?:["\']|\[)?([^"\';\n\]]+)(?:["\']|\])?', re.IGNORECASE)
        ],
        "method name  / shortname": [
            re.compile(r'method[_\s]*name[_\s*/]*shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'"Method name / Shortname"\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE),
            re.compile(r'Method\s*Name\s*/\s*Shortname\s*[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
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



