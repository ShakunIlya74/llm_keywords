import os
import time

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
    # Step 1: Retrieve Paper IDs
    paper_ids_text_pairs = load_abstracts(n_papers, paper_ids_text_pairs_path)
    # check if output file already exists
    output_dict = {}
    if os.path.exists(output_path):
        print(f"Output file already exists at {output_path}. Continuing from where it left off.")
        paper_summaries = read_dict(output_path)
        processed_paper_ids = set(paper_summaries.keys())
        paper_ids_to_process = set([paper_id for paper_id, _ in paper_ids_text_pairs]) - processed_paper_ids
        print(f"Processed {len(processed_paper_ids)} papers. {len(paper_ids_to_process)} papers left to process.")
        paper_ids_text_pairs = [(paper_id, text) for paper_id, text in paper_ids_text_pairs if paper_id in paper_ids_to_process]
        output_dict = paper_summaries

    # Step 3: Initialize Transformer Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # filter out None params
    # model_params = {k: v for k, v in model_params.items() if v is not None}

    for i, (paper_id, text) in enumerate(paper_ids_text_pairs):
        if i % 10 == 0:
            print(f"Processing paper {i + 1} of {len(paper_ids_text_pairs)}. Percent complete: {100 * (i + 1) / len(paper_ids_text_pairs):.2f}%")
        prompt = prompt_fn(text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Prepare the prompt using the chat template
        combined_prompt = ""
        for message in messages:
            if message["role"] == "system":
                combined_prompt += f"{message['content']}\n"
            elif message["role"] == "user":
                combined_prompt += f"{message['content']}\n"

        # Tokenize the prompt
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

        # Store the result
        output_dict[paper_id] = response

        print(f"Paper ID: {paper_id}")
        print(f"Abstract: {text}")
        print(f"{response}")
        print(f"Time taken: {toc - tic:.2f} seconds\n")

        if (i + 1) % checkpoint_freq == 0:
            # Save the output dictionary to a pickle file
            write_dict_to_pkl(output_dict, output_path)
    write_dict_to_pkl(output_dict, output_path)

    return output_dict



if __name__ == '__main__':
    summaries = query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct",
                                                 prompt_fn=prompt_with_text,
                                                 output_path="../data/llm_outputs/llm_summaries_test.pkl",
                                                 checkpoint_freq=1)


    # load and parse the summaries
    # parse_and_save_llm_outputs("../data/llm_outputs/llm_summaries_transformers.pkl", "../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # loaded_summaries = read_dict_from_pkl("../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # for paper_id, summary in loaded_summaries.items():
    #     print(f"Paper ID: {paper_id}")
    #     print(f"Summary: {summary}")
    #     print("\n")


