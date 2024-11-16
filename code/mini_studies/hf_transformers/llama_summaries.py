import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from code.mini_studies.hf_transformers.summary_utils import load_abstracts, write_dict_to_pkl


def query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct"):
    # Step 1: Retrieve Paper IDs
    paper_ids_text_pairs = load_abstracts(n_papers)

    # Step 3: Initialize Transformer Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_dict = {}

    for paper_id, text in paper_ids_text_pairs:
        prompt = f"""
Task: Based on the abstract provided, extract and label the following key details. Follow the structure exactly, keeping answers brief and specific. 
    Adhere strictly to the format.
    If any information is unclear or unavailable in the abstract, write "None." for that field.
    Use the exact labels and formatting provided. Do not include comments or repeat the response.
Details to Extract:
    field_of_Paper: * The primary academic discipline. * [insert answer]
    subfield: * The main research category within the field. * [insert answer]
    sub_subfield: * A narrower focus within the subfield. * [insert answer]
    keywords: * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
    method_name_shortname: * The main technique or model name proposed in the abstract. * [insert answer]
Abstract:
    '{text}'
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Prepare the prompt using the chat template
        # Note: `apply_chat_template` is not a standard method in Hugging Face Transformers.
        # Instead, we'll concatenate the messages to form the input prompt.
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
                max_new_tokens=512,
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
        print(f"{response}")
        print(f"Time taken: {toc - tic:.2f} seconds\n")

        # Save the output dictionary to a pickle file
        write_dict_to_pkl(output_dict, "../data/llm_outputs/llm_summaries_transformers.pkl")

    return output_dict

if __name__ == '__main__':
    summaries = query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct")
    # Optionally, you can load the summaries later using read_dict_from_pkl
    # loaded_summaries = read_dict_from_pkl("../data/variables/llm_outputs/llm_summaries_transformers.pkl")


