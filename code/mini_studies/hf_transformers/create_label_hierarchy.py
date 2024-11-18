from code.mini_studies.hf_transformers.summary_utils import parse_and_save_llm_outputs, read_dict_from_pkl, \
    write_dict_to_pkl


#             "field of paper": None,
#             "subfield": None,
#             "sub subfield": None,
#             "keywords": None,
#             "method name  / shortname": None,



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











if __name__ == '__main__':
    # parse_and_save_llm_outputs("../data/llm_outputs/llm_summaries_transformers_temp.pkl", "../data/llm_outputs/llm_summaries_transformers_parsed_temp.pkl")
    # get_sorted_occurrence_dict("../data/llm_outputs/llm_summaries_transformers_parsed_temp.pkl")

    output = parse_hierarchy_from_top_labels("../data/llm_outputs/llm_summaries_transformers_parsed_temp.pkl", top_fields=20, percent_of_top_subfields=0.3)
    print(len(output["Computer Science"]["subfields"].keys()), [(subfield, output["Computer Science"]["subfields"][subfield]["number_of_papers"]) for subfield in output["Computer Science"]["subfields"].keys()])
    # field_of_paper_occurrences = read_dict_from_pkl("../data/llm_outputs/statistics/field_of_paper_occurrences.pkl")
    # subfield_occurrences = read_dict_from_pkl("../data/llm_outputs/statistics/subfield_occurrences.pkl")
    # sub_subfield_occurrences = read_dict_from_pkl("../data/llm_outputs/statistics/sub_subfield_occurrences.pkl")
    # keywords_occurrences = read_dict_from_pkl("../data/llm_outputs/statistics/keywords_occurrences.pkl")
    # method_name_occurrences = read_dict_from_pkl("../data/llm_outputs/statistics/method_name_occurrences.pkl")
    # print("field_of_paper_occurrences:", list(field_of_paper_occurrences.items())[:20])


    # print("subfield_occurrences:", subfield_occurrences)
    # print("sub_subfield_occurrences:", sub_subfield_occurrences)