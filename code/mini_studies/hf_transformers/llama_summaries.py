import os
import time
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from code.mini_studies.hf_transformers.summary_utils import load_abstracts, write_dict_to_pkl, \
    parse_and_save_llm_outputs, read_dict_from_pkl, create_dirs_if_not_exist, read_dict


def prompt_with_text(text):
    prompt = f"""
    Task: Based on the abstract provided, extract and label the following key details. Follow the structure exactly, keeping answers brief and specific. 
        Adhere strictly to the format.
        If any information is unclear or unavailable in the abstract, write "None." for that field.
        Use the exact labels and formatting provided. Do not include comments. Do not repeat the response.
    Details to Extract:
        field_of_Paper = * The primary academic discipline. * [insert answer]
        subfield = * The main research category within the field. * [insert answer]
        sub_subfield = * A narrower focus within the subfield. * [insert answer]
        keywords = * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
        method_name_shortname = * The main technique or model name proposed in the abstract. * [insert answer]
    Abstract:
    '{text}'
    """
    return prompt


def new_prompt_with_text(text):
    prompt = f"""
    Task: Based on the abstract provided, extract and label the following key details. Follow the structure exactly, keeping answers brief and specific. 
        Adhere strictly to the format.
        If any information is unclear or unavailable in the abstract, write "None." for that field.
        Use the exact labels and formatting provided. Do not include comments. Do not repeat the response.
    Details to Extract:
        field_of_Paper = * The primary academic discipline. * [insert answer]
        subfield = * The main research category within the field. * [insert answer]
        sub_subfield = * A narrower focus within the subfield. * [insert answer]
        keywords = * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
        method_name_shortname = * The main technique or model name proposed in the abstract. * [insert answer]
    Abstract:
    '{text}'
    """
    return prompt


def query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct", prompt_fn=prompt_with_text,
                                     paper_ids_text_pairs_path="../data/llm_inputs/paper_ids_text_pairs.pkl",
                                     output_path="../data/llm_outputs/llm_summaries.pkl",
                                     checkpoint_freq=10,
                                     model_params = {
                                            "max_new_tokens": 512,
                                            "temperature" : None,
                                            "top_k": None,
                                            "top_p": None,
                                            "repetition_penalty": None,
                                            "do_sample": False}
                                     ):
    """
    Query a transformer model to generate summaries(labels+keywords) for a given set of papers.
    Args:
        n_papers:
        model_name:
        prompt_fn:
        paper_ids_text_pairs_path:
        output_path:
        checkpoint_freq:
        model_params: dict containing model parameters, default for the Qwen2.5-7B-Instruct model are
        temperature=0.7, top_k=20, top_p=0.8, do_sample=True

    Returns:

    """
    create_dirs_if_not_exist()
    paper_ids_text_pairs = load_abstracts(n_papers, paper_ids_text_pairs_path)
    # check if output file already exists
    output_dict = {}
    if os.path.exists(output_path):
        print(f"Output file already exists at {output_path}. Continuing from where it left off.")
        paper_summaries = read_dict(output_path)
        processed_paper_ids = set(paper_summaries.keys())
        paper_ids_text_pairs = [(paper_id, text) for paper_id, text in paper_ids_text_pairs if paper_id not in processed_paper_ids]
        output_dict = paper_summaries

    # Initialize the model and tokenizer, ensuring model is on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"  # removed device_map to avoid offloading
    )
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, (paper_id, text) in enumerate(paper_ids_text_pairs):
        if i % 10 == 0:
            print(f"Processing paper {i + 1} of {len(paper_ids_text_pairs)}. Percent complete: {100 * (i + 1) / len(paper_ids_text_pairs):.2f}%")
        prompt = prompt_fn(text)

        # Prepare the combined prompt
        combined_prompt = f"You are a helpful assistant.\n{prompt}\n"
        model_inputs = tokenizer([combined_prompt], return_tensors="pt").to(device)

        # Generate the response with default parameters
        tic = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **model_params
            )
        toc = time.time()

        # Decode the generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        output_dict[paper_id] = response

        print(f"Paper ID: {paper_id}\nAbstract: {text}\n{response}\nTime taken: {toc - tic:.2f} seconds\n")

        if (i + 1) % checkpoint_freq == 0:
            write_dict_to_pkl(output_dict, output_path)
    write_dict_to_pkl(output_dict, output_path)

    # Optionally free model and tokenizer memory at the end of the function
    del model, tokenizer
    torch.cuda.empty_cache()

    return output_dict


def compare_two_runs(n_papers,
                     model_params1, model_params2,
                     prompt_fn1, prompt_fn2,
                     output_path1, output_path2,
                     model_name="Qwen/Qwen2.5-7B-Instruct",
                     paper_ids_text_pairs_path="../data/llm_inputs/paper_ids_text_pairs.pkl",
                     checkpoint_freq=10,
                     run_inference=True):
    """
    Compare two runs of query_transformers_for_summaries on the same set of papers.

    Args:
        n_papers (int): Number of papers to process.
        model_params1 (dict): Model parameters for the first run.
        model_params2 (dict): Model parameters for the second run.
        prompt_fn1 (callable): Prompt function for the first run.
        prompt_fn2 (callable): Prompt function for the second run.
        output_path1 (str): Path to save outputs for the first run.
        output_path2 (str): Path to save outputs for the second run.
        model_name (str): Name of the model to use.
        paper_ids_text_pairs_path (str): Path to the pickle file containing paper IDs and texts.
        checkpoint_freq (int): Frequency of checkpoints when processing.
        run_inference (bool): Whether to run inference or not. If not - only comparison on saved outputs will be done.

    Returns:
        None. The function saves two output files and prints a comparison of results.
    """
    if run_inference:
        # Run 1: Using prompt_fn1 and model_params1
        print("Starting Run 1...")
        start_time_run1 = time.time()
        outputs1 = query_transformers_for_summaries(
            n_papers=n_papers,
            model_name=model_name,
            prompt_fn=prompt_fn1,
            paper_ids_text_pairs_path=paper_ids_text_pairs_path,
            output_path=output_path1,
            checkpoint_freq=checkpoint_freq,
            model_params=model_params1
        )
        end_time_run1 = time.time()

        gc.collect()
        torch.cuda.empty_cache()
        print("Cleared GPU resources after Run 1.")

        # Run 2: Using prompt_fn2 and model_params2
        print("\nStarting Run 2...")
        start_time_run2 = time.time()
        outputs2 = query_transformers_for_summaries(
            n_papers=n_papers,
            model_name=model_name,
            prompt_fn=prompt_fn2,
            paper_ids_text_pairs_path=paper_ids_text_pairs_path,
            output_path=output_path2,
            checkpoint_freq=checkpoint_freq,
            model_params=model_params2
        )
        end_time_run2 = time.time()
        print(f"Run 1 completed in {end_time_run1 - start_time_run1:.2f} seconds")
        print(f"Run 2 completed in {end_time_run2 - start_time_run2:.2f} seconds")

    abstracts_dict = dict(load_abstracts(n_papers, paper_ids_text_pairs_path))

    parsed_output1 = parse_and_save_llm_outputs(output_path1,
                               output_path1+"_parsed.pkl")
    parsed_output2 = parse_and_save_llm_outputs(output_path2,
                                 output_path2+"_parsed.pkl")


    # Compare outputs for each paper by printing them side by side
    print("\nComparing outputs for each paper:")
    all_paper_ids = set(parsed_output1.keys()).union(parsed_output2.keys())
    differences = {
            "field of paper": 0,
            "subfield": 0,
            "sub subfield": 0,
            "keywords": 0,
            "method name  / shortname": 0,
        }
    for paper_id in sorted(all_paper_ids):
        result1 = parsed_output1.get(paper_id, {})
        result2 = parsed_output2.get(paper_id, {})
        diff_found = False
        # compare each field in dict and print if different
        for key in differences.keys():
            if result1.get(key) != result2.get(key):
                differences[key] += 1
                if not diff_found:
                    print(f"Paper ID: {paper_id}")
                    print(f"Abstract: {abstracts_dict.get(paper_id)}")
                    diff_found = True
                print(f"{key}:\nRun 1: {result1.get(key)} \nRun 2: {result2.get(key)}")
        if diff_found:
            print("---------------------\n")
    print("\nSummary of differences:")
    for key, value in differences.items():
        print(f"{key}: {value} papers with different values.")
    print("\n")



def run_comparison(n_papers=10, delete_existing_outputs=False, run_inference=True):

    model_params1 = {
        "max_new_tokens": 512,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
        "do_sample": False}
    model_params2 = {
        "max_new_tokens": 512,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.1,
        "do_sample": False}

    prompt_fn1 = prompt_with_text
    prompt_fn2 = prompt_with_text

    output_path1 = "../data/llm_outputs/llm_summaries_test1.pkl"
    output_path2 = "../data/llm_outputs/llm_summaries_test2.pkl"

    if delete_existing_outputs:
        for output_path in [output_path1, output_path2]:
            if os.path.exists(output_path):
                os.remove(output_path)

    compare_two_runs(n_papers, model_params1, model_params2, prompt_fn1, prompt_fn2, output_path1, output_path2,
                     run_inference=run_inference)




if __name__ == '__main__':
    # summaries = query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct",
    #                                              prompt_fn=prompt_with_text,
    #                                              output_path="../data/llm_outputs/llm_summaries_test.pkl",
    #                                              checkpoint_freq=1)

    run_comparison(n_papers=100, delete_existing_outputs=True, run_inference=True)


    # load and parse the summaries
    # parse_and_save_llm_outputs("../data/llm_outputs/llm_summaries_transformers.pkl", "../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # loaded_summaries = read_dict_from_pkl("../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # for paper_id, summary in loaded_summaries.items():
    #     print(f"Paper ID: {paper_id}")
    #     print(f"Summary: {summary}")
    #     print("\n")


