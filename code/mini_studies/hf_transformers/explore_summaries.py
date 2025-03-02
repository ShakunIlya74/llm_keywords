import pandas as pd

from code.mini_studies.hf_transformers.llama_summaries import given_fields_list
from code.mini_studies.hf_transformers.summary_utils import parse_and_save_llm_outputs, read_dict_from_pkl





def check_summaries(row_summaries_path="../data/llm_outputs/llm_summaries_pretty20_fields.pkl",
                    parsed_summaries_output="../data/llm_outputs/parsed_summaries/llm_summaries_20_fields_parsed.pkl"):
    parse_and_save_llm_outputs(row_summaries_path,
                               parsed_summaries_output)
    loaded_summaries = read_dict_from_pkl(parsed_summaries_output)
    print(f"Loading {len(loaded_summaries)} summaries")

    # Initialize lists to collect data for the DataFrames
    field_data = []
    subfield_data = []
    sub_subfield_data = []

    # Collect occurrences
    for paper_id, details in loaded_summaries.items():
        field = details.get("field of paper", "Unknown")
        subfield = details.get("subfield", "Unknown")
        sub_subfield = details.get("sub subfield", "Unknown")

        field_data.append((paper_id, field))
        subfield_data.append((paper_id, subfield))
        sub_subfield_data.append((paper_id, sub_subfield))

    # Create DataFrames
    field_df = pd.DataFrame(field_data, columns=["paper_id", "field"])
    subfield_df = pd.DataFrame(subfield_data, columns=["paper_id", "subfield"])
    sub_subfield_df = pd.DataFrame(sub_subfield_data, columns=["paper_id", "sub_subfield"])

    # Group and count occurrences
    total_papers = len(loaded_summaries)
    field_counts = field_df.groupby("field").size().reset_index(name="number_of_papers")
    subfield_counts = subfield_df.groupby("subfield").size().reset_index(name="number_of_papers")
    sub_subfield_counts = sub_subfield_df.groupby("sub_subfield").size().reset_index(name="number_of_papers")

    # Sort counts in descending order
    field_counts = field_counts.sort_values(by="number_of_papers", ascending=False)
    subfield_counts = subfield_counts.sort_values(by="number_of_papers", ascending=False)
    sub_subfield_counts = sub_subfield_counts.sort_values(by="number_of_papers", ascending=False)
    print("Field counts:")
    print(field_counts)
    # print fields and their counts which are not in the given fields list
    print("Fields not in the given fields list:")
    print(field_counts[~field_counts["field"].isin(given_fields_list)])
    # percent of papers with fields not in the given fields list
    print("Percentage of papers with fields not in the given fields list:")
    print(f'{round(len(field_counts[~field_counts["field"].isin(given_fields_list)]) / total_papers * 100, 2)}%')



if __name__ == "__main__":
    check_summaries()