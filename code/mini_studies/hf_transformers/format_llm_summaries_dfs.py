import pandas as pd

from code.mini_studies.hf_transformers.llama_summaries import given_fields_list
from code.mini_studies.hf_transformers.summary_utils import read_dict_from_pkl
from pathlib import Path

SI_LABEL_DFS_DIR = Path("../../scholar_inbox/data/scholar_map/llm_outputs/summaries_dfs")

def filter_and_parse_dict_summaries_to_df(summaries_dict_path, save_to_path,
                                          fields_list=given_fields_list):
    """
    Parse the summaries dictionary into a DataFrame for easier manipulation
    and filter rows that are not in the given fields list.
    """
    # load the summaries dictionary
    summaries_dict = read_dict_from_pkl(summaries_dict_path)
    print(f"Loaded {len(summaries_dict)} summaries.")

    # Initialize lists to collect data for the DataFrames
    l1_list = []
    l2_list = []
    l3_list = []
    keywords_list = []
    method_name_list = []

    # Collect occurrences
    for paper_id, details in summaries_dict.items():
        field = details.get("field of paper", "Unknown")
        subfield = details.get("subfield", "Unknown")
        sub_subfield = details.get("sub subfield", "Unknown")
        keywords = details.get("keywords", [])
        method_name = details.get("method name  / shortname", "Unknown")

        l1_list.append((paper_id, field))
        l2_list.append((paper_id, subfield))
        l3_list.append((paper_id, sub_subfield))
        keywords_list.append((paper_id, keywords))
        method_name_list.append((paper_id, method_name))

    # Create DataFrames
    df = pd.DataFrame(l1_list, columns=["paper_id", "l1_pure"])
    # add columns to the DataFrame
    df["l2_pure"] = pd.DataFrame(l2_list, columns=["paper_id", "l2"])["l2"]
    df["l3_pure"] = pd.DataFrame(l3_list, columns=["paper_id", "l3"])["l3"]
    df["keywords_pure"] = pd.DataFrame(keywords_list, columns=["paper_id", "keywords"])["keywords"]
    df["method_name_pure"] = pd.DataFrame(method_name_list, columns=["paper_id", "method_name"])["method_name"]

    # log the number of rows in the DataFrame
    print(f"Created DataFrame with {len(df)} rows.")
    # filter rows that are not in the given fields list
    if fields_list:
        df = df[df["l1_pure"].isin(fields_list)]
        print(f"Filtered DataFrame to {len(df)} rows with fields in the given fields list.")

    # save the DataFrame
    df["paper_id"] = df["paper_id"].astype("int64")
    df.to_parquet(save_to_path, index=False, engine="fastparquet")
    print(f"Saved DataFrame to {save_to_path}.")
    return save_to_path



if __name__ == "__main__":
    # filter_and_parse_dict_summaries_to_df("../data/llm_outputs/llm_summaries_transformers_parsed.pkl",
    #                            Path(SI_LABEL_DFS_DIR, "cache_labels_df.parquet"))

    filter_and_parse_dict_summaries_to_df("../data/llm_outputs/parsed_summaries/llm_summaries_20_fields_parsed.pkl",
                                          Path(SI_LABEL_DFS_DIR, "top200k_v2_labels_df.parquet"))