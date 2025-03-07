import json
import pickle
from pathlib import Path

import numpy as np

from code.mini_studies.hf_transformers.summary_utils import parse_and_save_llm_outputs, read_dict_from_pkl, \
    write_dict_to_pkl


#             "field of paper": None,
#             "subfield": None,
#             "sub subfield": None,
#             "keywords": None,
#             "method name  / shortname": None,

SI_VARS_PATH = Path("../../scholar_inbox/data/scholar_map/variables")



def get_sorted_occurrence_dict(path_to_parsed_llm_outputs):
    loaded_summaries = read_dict_from_pkl(path_to_parsed_llm_outputs)
    print(f"Loading {len(loaded_summaries)} summaries")
    field_of_paper_occurrences = {}
    subfield_occurrences = {}
    sub_subfield_occurrences = {}
    keywords_occurrences = {}
    method_name_occurrences = {}
    for key, value in loaded_summaries.items():
        field_of_paper = value["field of paper"]
        subfield = value["subfield"]
        sub_subfield = value["sub subfield"]
        keywords = value["keywords"]
        method_name = value["method name  / shortname"]
        if field_of_paper in field_of_paper_occurrences:
            field_of_paper_occurrences[field_of_paper] += 1
        else:
            field_of_paper_occurrences[field_of_paper] = 1
        if subfield in subfield_occurrences:
            subfield_occurrences[subfield] += 1
        else:
            subfield_occurrences[subfield] = 1
        if sub_subfield in sub_subfield_occurrences:
            sub_subfield_occurrences[sub_subfield] += 1
        else:
            sub_subfield_occurrences[sub_subfield] = 1
        for keyword in keywords:
            if keyword in keywords_occurrences:
                keywords_occurrences[keyword] += 1
            else:
                keywords_occurrences[keyword] = 1
        if method_name in method_name_occurrences:
            method_name_occurrences[method_name] += 1
        else:
            method_name_occurrences[method_name] = 1

    sorted_field_of_paper_occurrences = {k: v for k, v in sorted(field_of_paper_occurrences.items(), key=lambda item: item[1], reverse=True)}
    sorted_subfield_occurrences = {k: v for k, v in sorted(subfield_occurrences.items(), key=lambda item: item[1], reverse=True)}
    sorted_sub_subfield_occurrences = {k: v for k, v in sorted(sub_subfield_occurrences.items(), key=lambda item: item[1], reverse=True)}
    sorted_keywords_occurrences = {k: v for k, v in sorted(keywords_occurrences.items(), key=lambda item: item[1], reverse=True)}
    sorted_method_name_occurrences = {k: v for k, v in sorted(method_name_occurrences.items(), key=lambda item: item[1], reverse=True)}
    # save dicts to files
    write_dict_to_pkl(sorted_field_of_paper_occurrences, "../data/llm_outputs/statistics/field_of_paper_occurrences.pkl")
    write_dict_to_pkl(sorted_subfield_occurrences, "../data/llm_outputs/statistics/subfield_occurrences.pkl")
    write_dict_to_pkl(sorted_sub_subfield_occurrences, "../data/llm_outputs/statistics/sub_subfield_occurrences.pkl")
    write_dict_to_pkl(sorted_keywords_occurrences, "../data/llm_outputs/statistics/keywords_occurrences.pkl")
    write_dict_to_pkl(sorted_method_name_occurrences, "../data/llm_outputs/statistics/method_name_occurrences.pkl")
    return sorted_field_of_paper_occurrences, sorted_subfield_occurrences, sorted_sub_subfield_occurrences, sorted_keywords_occurrences, sorted_method_name_occurrences




def parse_hierarchy_from_top_labels(path_to_parsed_llm_outputs, top_fields=20, percent_of_top_subfields=0.3):
    loaded_summaries = read_dict_from_pkl(path_to_parsed_llm_outputs)
    print(f"Loading {len(loaded_summaries)} summaries")
    field_of_paper_occurrences = {}
    for paper_id, summary_dict in loaded_summaries.items():
        field_of_paper = summary_dict["field of paper"]
        if field_of_paper in field_of_paper_occurrences:
            field_of_paper_occurrences[field_of_paper] += 1
        else:
            field_of_paper_occurrences[field_of_paper] = 1
    sorted_field_of_paper_occurrences = {k: v for k, v in sorted(field_of_paper_occurrences.items(), key=lambda item: item[1], reverse=True)}
    # get list of top fields
    top_fields_list = list(sorted_field_of_paper_occurrences.keys())[:top_fields]
    field_hierarchy = {field : {"number_of_papers": sorted_field_of_paper_occurrences[field], "summaries": {}, "subfields": {}} for field in top_fields_list}
    for paper_id, summary_dict in loaded_summaries.items():
        field_of_paper = summary_dict["field of paper"]
        subfield = summary_dict["subfield"]
        sub_subfield = summary_dict["sub subfield"]
        keywords = summary_dict["keywords"]
        method_name = summary_dict["method name  / shortname"]
        if field_of_paper in top_fields_list:
            field_hierarchy[field_of_paper]["summaries"][paper_id] = summary_dict
            if subfield not in field_hierarchy[field_of_paper]["subfields"]:
                field_hierarchy[field_of_paper]["subfields"][subfield] = {"number_of_papers": 1, "summaries": {paper_id: summary_dict}}
            else:
                field_hierarchy[field_of_paper]["subfields"][subfield]["number_of_papers"] += 1
                field_hierarchy[field_of_paper]["subfields"][subfield]["summaries"][paper_id] = summary_dict
    for field in top_fields_list:
        subfields = field_hierarchy[field]["subfields"]
        sorted_subfields = {k: v for k, v in sorted(subfields.items(), key=lambda item: item[1]["number_of_papers"], reverse=True)}
        top_subfields = list(sorted_subfields.keys())[:int(percent_of_top_subfields * len(sorted_subfields))]
        field_hierarchy[field]["subfields"] = {subfield : subfields[subfield] for subfield in top_subfields}
    # save dict to file
    write_dict_to_pkl(field_hierarchy, f"../data/llm_outputs/statistics/field_hierarchy_basic_temp.pkl")
    return field_hierarchy


# def create_label_layers(llm_summaries_path, output_dir):
#     loaded_summaries = read_dict_from_pkl(llm_summaries_path)
#     print(f"Loading {len(loaded_summaries)} summaries")
#     # write keys to the pickle file
#     with open(output_dir+'paper_ids_for_labels.pkl', 'wb') as f:
#         pickle.dump(loaded_summaries.keys(), f)
#     fields, subfields, sub_subfields, keywords, method_names = get_sorted_occurrence_dict(llm_summaries_path)
#
#     # create pd df with paper_ids, fields, number of papers


import pandas as pd
import pickle
from collections import Counter

import pandas as pd
import pickle
from collections import Counter


import numpy as np
import pandas as pd

def create_label_layers(llm_summaries_path, output_dir='../data/llm_outputs/keywords/',
                        coverage_percentage_l1=0.2, coverage_percentage_l2=0.2):
    """
    Creates label layers for fields, subfields, and sub-subfields, filtering subfields and sub-subfields
    to retain only the top labels that collectively account for at least 20% of the total papers.
    """
    # Load summaries
    loaded_summaries = read_dict_from_pkl(llm_summaries_path)
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

    # Determine top subfields and sub-subfields that cover at least 20% of papers
    def get_top_labels(counts_df, total_papers, coverage_percentage):
        cumulative_sum = 0
        selected_labels = []
        coverage_threshold = total_papers * coverage_percentage
        for _, row in counts_df.iterrows():
            cumulative_sum += row["number_of_papers"]
            selected_labels.append(row["subfield"] if "subfield" in counts_df.columns else row["sub_subfield"])
            if cumulative_sum >= coverage_threshold:
                break
        return selected_labels

    np.save(output_dir + 'field_layer_random_cache_pure', field_df["field"].values, allow_pickle=True)
    np.save(output_dir + 'subfield_layer_random_cache_pure', subfield_df["subfield"].values, allow_pickle=True)
    np.save(output_dir + 'sub_subfield_layer_random_cache_pure', sub_subfield_df["sub_subfield"].values, allow_pickle=True)
    # determine the top n fields
    top_fields = field_counts["field"].values[:20]
    # determine the top subfields and sub-subfields
    top_subfields = get_top_labels(subfield_counts, total_papers, coverage_percentage_l1)
    top_sub_subfields = get_top_labels(sub_subfield_counts, total_papers, coverage_percentage_l2)

    # Map unpopular labels to 'None'
    field_df["field"] = field_df["field"].apply(lambda x: x if x in top_fields else "None")
    subfield_df["subfield"] = subfield_df["subfield"].apply(lambda x: x if x in top_subfields else "None")
    sub_subfield_df["sub_subfield"] = sub_subfield_df["sub_subfield"].apply(
        lambda x: x if x in top_sub_subfields else "None")

    # Save processed layers to files
    np.save(output_dir + 'field_layer_random_cache', field_df["field"].values, allow_pickle=True)
    np.save(output_dir + 'subfield_layer_random_cache', subfield_df["subfield"].values, allow_pickle=True)
    np.save(output_dir + 'sub_subfield_layer_random_cache', sub_subfield_df["sub_subfield"].values, allow_pickle=True)
    np.save(output_dir + 'paper_ids_random_cache', field_df["paper_id"].values)

    # Print summary statistics
    print("Number of unique values in each layer:")
    print("Field:", len(field_df["field"].unique()))
    print("Subfield:", len(subfield_df["subfield"].unique()))
    print("Sub-subfield:", len(sub_subfield_df["sub_subfield"].unique()))



def filter_repeating_subfields():
    """
    Replaces repeating subfields with 'None' and prints the number of substitutions made.
    """
    # Load layers
    label_layer1 = np.load(SI_VARS_PATH / "field_layer_random_cache.npy", allow_pickle=True)
    label_layer2 = np.load(SI_VARS_PATH / "subfield_layer_random_cache.npy", allow_pickle=True)
    label_layer3 = np.load(SI_VARS_PATH / "sub_subfield_layer_random_cache.npy", allow_pickle=True)

    # Ensure labels are in string format
    label_layer1 = label_layer1.astype(str)
    label_layer2 = label_layer2.astype(str)
    label_layer3 = label_layer3.astype(str)

    # Substitution masks
    matrix_for_labels1 = np.isin(label_layer2, label_layer1)
    matrix_for_labels2 = np.isin(label_layer3, label_layer2)

    # Count substitutions
    substitutions_layer2 = np.sum(matrix_for_labels1)
    substitutions_layer3 = np.sum(matrix_for_labels2)

    # Perform substitutions
    label_layer2[matrix_for_labels1] = 'None'
    label_layer3[matrix_for_labels2] = 'None'

    # replace 'None.' with 'None'
    label_layer2 = np.where(label_layer2 == 'None.', 'None', label_layer2)
    label_layer3 = np.where(label_layer3 == 'None.', 'None', label_layer3)

    # Save updated layers
    np.save(SI_VARS_PATH / "subfield_layer_random_cache", label_layer2)
    np.save(SI_VARS_PATH / "sub_subfield_layer_random_cache", label_layer3)

    # Print debug information
    print("Number of substitutions made:")
    print(f"Subfield layer: {substitutions_layer2}")
    print(f"Sub-subfield layer: {substitutions_layer3}")

    print("\nNumber of unique values in each layer after substitution:")
    print("Field:", len(np.unique(label_layer1)))
    print("Subfield:", len(np.unique(label_layer2)))
    print("Sub-subfield:", len(np.unique(label_layer3)))

    # Check specific example
    print("\nDebug checks:")
    print("Natural Language Processing in Field:", "Natural Language Processing" in label_layer1)
    print("Natural Language Processing in Subfield:", "Natural Language Processing" in label_layer2)
    print("Natural Language Processing in Sub-subfield:", "Natural Language Processing" in label_layer3)


if __name__ == '__main__':
    # parse_and_save_llm_outputs("../data/llm_outputs/llm_summaries_transformers.pkl", "../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # get_sorted_occurrence_dict("../data/llm_outputs/llm_summaries_transformers_parsed_temp.pkl")
    # outputs = read_dict_from_pkl("../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # print(len(outputs))
    # # filter all from Computer science field
    # outputs = {key: value for key, value in outputs.items() if value["field of paper"] == "Computer Science"}
    # # filter firt 100
    # outputs = dict(list(outputs.items())[:100])
    # print(len(outputs))
    # # save to new json file
    # with open("../../scholar_inbox/data/scholar_map/llm_outputs/llm_output_cs_sample.json", "w") as f:
    #     json.dump(outputs, f)

    # output = parse_hierarchy_from_top_labels("../data/llm_outputs/llm_summaries_transformers_parsed.pkl", top_fields=20, percent_of_top_subfields=0.3)
    # print(len(output["Computer Science"]["subfields"].keys()), [(subfield, output["Computer Science"]["subfields"][subfield]["number_of_papers"]) for subfield in output["Computer Science"]["subfields"].keys()])

    create_label_layers("../data/llm_outputs/llm_summaries_transformers_parsed.pkl",
                        output_dir='../../scholar_inbox/data/scholar_map/variables/',
                        coverage_percentage_l1=0.65, coverage_percentage_l2=0.6)
    filter_repeating_subfields()


    # label_layer1 = np.load(SI_VARS_PATH / "field_layer_random_cache.npy", allow_pickle=True)
    # label_layer2 = np.load(SI_VARS_PATH / "subfield_layer_random_cache.npy", allow_pickle=True)
    # label_layer3 = np.load(SI_VARS_PATH / "sub_subfield_layer_random_cache.npy", allow_pickle=True)
    #
    # # check if 'Natural Language Processing' is in the subfield layer
    # print("Natural Language Processing" in label_layer1)
    # print("Natural Language Processing" in label_layer2)
    # print("Natural Language Processing" in label_layer3)







    pass