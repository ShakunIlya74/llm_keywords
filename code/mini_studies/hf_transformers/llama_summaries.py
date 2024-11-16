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
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_dict = {}

    for paper_id, text in paper_ids_text_pairs:
        prompt = f"""
                Given the abstract below, please extract and label the following key details clearly. Each response should be straightforward and specific:
                    field_of_Paper = * The primary academic discipline * [insert answer];
                    subfield = * The main research category within the field * [insert answer];
                    sub_subfield = * A narrower focus within the subfield * [insert answer];
                    keywords = * A set of 3-5 words or phrases that describe the core topics. * [insert answer];
                    method_name_shortname = *The main technique or model name proposed in the abstract.* [insert answer];
                
                Abstract: '{text}'
                
                Ensure each label is followed by a concise answer with minimal formatting issues, and separate multiple keywords with commas. 
                If information is not clear from the abstract, write "N/A" for that field."""

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate the response
        tic = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
        toc = time.time()

        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # Store the result
        output_dict[paper_id] = generated_text

        print(f"Paper ID: {paper_id}")
        print(f"Output: {generated_text}")
        print(f"Time taken: {toc - tic:.2f} seconds\n")

        # Save the output dictionary to a pickle file
        write_dict_to_pkl(output_dict, "../data/llm_outputs/llm_summaries_transformers.pkl")

    return output_dict

if __name__ == '__main__':
    summaries = query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct")
    # Optionally, you can load the summaries later using read_dict_from_pkl
    # loaded_summaries = read_dict_from_pkl("../data/variables/llm_outputs/llm_summaries_transformers.pkl")


