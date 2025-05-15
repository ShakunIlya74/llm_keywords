import os
import time
import gc

from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

from code.mini_studies.hf_transformers.summary_utils import load_texts, write_dict_to_pkl, \
    parse_and_save_llm_outputs, read_dict_from_pkl, create_dirs_if_not_exist, read_dict


given_fields_list = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Computer Science",
    "Electrical Engineering",
    "Engineering",
    "Materials Science",
    "Astronomy",
    "Earth Science",
    "Biology",
    "Medicine",
    "Economics",
    "Political Science",
    "Sociology",
    "Psychology",
    "Linguistics",
    "Philosophy",
    "History",
    "Geography",
    "Arts"
]


def prompt_with_text(title, abstract):
    # combine text
    text = title + "\n" + abstract
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


def new_prompt_with_text(title, abstract):
    # combine text
    text = title + "\n" + abstract
    prompt = f"""
    Task: Based on the abstract provided, extract and label the following key details. Follow the structure exactly, keeping answers brief and specific.
        Adhere strictly to the format.
        If any information is unclear or unavailable in the abstract, write "None." for that field.
        Use the exact labels and formatting provided. Do not include comments. Do not repeat the response.

    Note: For the field_of_Paper, choose one from the following list of academic disciplines:
        Mathematics
        Physics
        Chemistry
        Computer Science
        Electrical Engineering
        Engineering
        Materials Science
        Astronomy
        Earth Science
        Biology
        Medicine
        Economics
        Political Science
        Sociology
        Psychology
        Linguistics
        Philosophy
        History
        Geography
        Arts

    Details to Extract:
        field_of_Paper = * The primary academic discipline from the list above. * [insert answer]
        subfield = * The main research category within the field. * [insert answer]
        sub_subfield = * A narrower focus within the subfield. * [insert answer]
        keywords = * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
        method_name_shortname = * The main technique or model name proposed in the abstract. * [insert answer]

    Abstract:
    '{text}'
    """
    return prompt

def new_prompt_with_text_v2_0(title, abstract):
    prompt = f"""
Task: Based on the title and abstract provided, extract and label the following key details:
paper_field, paper_subfield, paper_sub_subfield, paper_keywords, paper_method_name.
Do not include any extra text, commentary, or repetition.

Note: For paper_field, choose one from the following list of academic disciplines:
Mathematics, Physics, Chemistry, Computer Science, Electrical Engineering, Engineering, Materials Science, Astronomy, Earth Science, Biology, Medicine, Economics, Political Science, Sociology, Psychology, Linguistics, Philosophy, History, Geography, Arts.

Instructions:
- Extract the relevant information for each field from the title and abstract. If any information is unclear or unavailable, write "None." exactly for that field.
- Follow the output format exactly as specified (do not add, omit, or repeat any parts of the format)

Format:
    paper_field = * What is the primary academic discipline from the list below? * [insert answer]
    paper_subfield = * What is the main research category within the paper_field? * [insert answer]
    paper_sub_subfield = * What is a narrower focus within the paper_subfield? * [insert answer]
    paper_keywords = * Which 3–5 keywords or phrases best capture the core topics of the paper? * [insert answer]
    paper_method_name = * Which single, most prominent technique or model is proposed in the abstract? * [insert answer]

Title: '{title}'
Abstract: '{abstract}'
    """
    return prompt


def new_prompt_with_text_v3_0(title, abstract):
    prompt = f"""
    Task: Based on the title and abstract provided, extract and label the following key details exactly as specified:
        field_of_Paper, subfield, sub_subfield, keywords, method_name_shortname.
    Follow the structure exactly and keep your answers brief and specific.
    Adhere strictly to the format. If any information is unclear or unavailable in the abstract, write "None." for that field.
    Use the exact labels and formatting provided. Do not include comments or repeat any part of the response.

    Note: For field_of_Paper, choose one from the following list of academic disciplines:
        Mathematics
        Physics
        Chemistry
        Computer Science
        Electrical Engineering
        Engineering
        Materials Science
        Astronomy
        Earth Science
        Biology
        Medicine
        Economics
        Political Science
        Sociology
        Psychology
        Linguistics
        Philosophy
        History
        Geography
        Arts

    Details to Extract:
        field_of_Paper = * The primary academic discipline from the list above. * [insert answer]
        subfield = * The main research category within the field. * [insert answer]
        sub_subfield = * A narrower focus within the subfield. * [insert answer]
        keywords = * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
        method_name_shortname = * The main technique or model name proposed in the abstract. * [insert answer]

    Title: '{title}'
    Abstract: '{abstract}'
    """

    return prompt

def new_prompt_with_text_v3_5(title, abstract):
    prompt = f"""
    Task: Based on the title and abstract provided, extract and label the following key details exactly as specified:
        paper_field, paper_subfield, paper_sub_subfield, paper_keywords, paper_method_name.
    Follow the structure exactly and keep your answers brief and specific.
    Adhere strictly to the format. If any information is unclear or unavailable in the abstract, write "None." for that field.
    Use the exact labels and formatting provided. Do not include comments or repeat any part of the response.

    Note: For paper_field, choose one from the following list of academic disciplines:
        Mathematics
        Physics
        Chemistry
        Computer Science
        Electrical Engineering
        Engineering
        Materials Science
        Astronomy
        Earth Science
        Biology
        Medicine
        Economics
        Political Science
        Sociology
        Psychology
        Linguistics
        Philosophy
        History
        Geography
        Arts

    Details to Extract:
        paper_field = * The primary academic discipline from the list above. * [insert answer]
        paper_subfield = * The main research category within the paper_field. * [insert answer]
        paper_sub_subfield = * A narrower focus within the paper_subfield. * [insert answer]
        paper_keywords = * A set of 3-5 words or phrases that describe the core topics, separated by commas. * [insert answer]
        paper_method_name_shortname = * The main technique or model name proposed in the paper. * [insert answer]

    Title: '{title}'
    Abstract: '{abstract}'
    """

    return prompt


def new_prompt_with_text_v4_0(title, abstract):
    prompt = f"""
    You are an information extraction assistant.  
    Given the following paper title and abstract, extract exactly these five fields and output _only_ a single valid JSON object (no explanations, no extra keys, no bullet points):
    
        1. "field_of_Paper"           – one of [Mathematics, Physics, Chemistry, Computer Science, Electrical Engineering, Engineering, Materials Science, Astronomy, Earth Science, Biology, Medicine, Economics, Political Science, Sociology, Psychology, Linguistics, Philosophy, History, Geography, Arts]
        2. "subfield"                 – the main research category within the field
        3. "sub_subfield"             – a narrower focus within the subfield
        4. "keywords"                 – an array of 3–5 phrases (strings) describing core topics
        5. "method_name_shortname"    – the main technique or model name proposed
    
    Use `null` (not an empty string) for any field you cannot identify.  
    **Your output must be valid JSON**. Example format:
    ```json
    {{
      "field_of_Paper": "Computer Science",
      "subfield": "Natural Language Processing",
      "sub_subfield": "Text Summarization",
      "keywords": ["summarization", "transformer", "fine-tuning"],
      "method_name_shortname": null
    }}
    Title: "{title}"
    Abstract: "{abstract}"
    """
    return prompt


def new_prompt_with_text_v4_1(title, abstract):
    prompt = f"""
    You are an expert information‐extraction assistant.  
    Your task is to read the paper title and abstract below, pull out exactly these five pieces of information, and output _only_ a single valid JSON object—no explanations, no extra keys, no comments:
    
      • field_of_Paper           (string or null)  
      • subfield                 (string or null)  
      • sub_subfield             (string or null)  
      • keywords                 (array of 3–5 strings or null)  
      • method_name_shortname    (string or null)  
    
    **Rules:**  
    1. Your output **must** be valid JSON.  
    2. Use the exact key names above (in quotes).  
    3. If you cannot identify a field, set its value to `null`.  
    4. Do _not_ include any other fields or text.  
    5. Choose `field_of_Paper` from exactly one of:
       ["Mathematics","Physics","Chemistry","Computer Science","Electrical Engineering",
        "Engineering","Materials Science","Astronomy","Earth Science","Biology",
        "Medicine","Economics","Political Science","Sociology","Psychology",
        "Linguistics","Philosophy","History","Geography","Arts"]
    
    **Output example**:  
    ```json
    {{
      "field_of_Paper": "Computer Science",
      "subfield": "Natural Language Processing",
      "sub_subfield": "Text Summarization",
      "keywords": ["summarization", "transformer", "fine-tuning"],
      "method_name_shortname": null
    }}
    Now process this paper:
    
    Title: "{title}"
    Abstract: "{abstract}"
    """
    return prompt


def new_prompt_with_text_v4_1_5(title, abstract):
    prompt = f"""
    You are an expert information‐extraction assistant.  
    Your task is to read the paper title and abstract below, pull out exactly these five pieces of information, and output _only_ a single valid JSON object—no explanations, no extra keys, no comments:

      • field_of_Paper           (string or null)  
      • subfield                 (string or null)  
      • sub_subfield             (string or null)  
      • keywords                 (array of 3–5 strings or null)  
      • method_name_shortname    (string or null)  

    **Rules:**  
    1. Your output **must** be valid JSON.  
    2. Use the exact key names above (in quotes).  
    3. If you cannot identify a field, set its value to `null`.  
    4. Do _not_ include any other fields or text.  
    5. Choose `field_of_Paper` from exactly one of:
       ["Mathematics","Physics","Chemistry","Computer Science","Electrical Engineering",
        "Engineering","Materials Science","Astronomy","Earth Science","Biology",
        "Medicine","Economics","Political Science","Sociology","Psychology",
        "Linguistics","Philosophy","History","Geography","Arts"]

    **Output json structure example**:  
    ```json
    {{
      "field_of_Paper": null,
      "subfield": null,
      "sub_subfield": null,
      "keywords": null,
      "method_name_shortname": null
    }}
    ```
    Now process this paper:

    Title: "{title}"
    Abstract: "{abstract}"
    """
    return prompt

def new_prompt_with_text_v4_2(title, abstract):
    prompt = f"""
    You are an expert information‐extraction assistant.
    
    Your task is to read the paper title and abstract below and extract exactly these five fields, outputting _only_ a single valid JSON object in this exact structure (replace the empty placeholders with real values or `null` if unavailable, and do not include any extra text or keys):
    
    ```json
    {{
      "field_of_Paper": "",
      "subfield": "",
      "sub_subfield": "",
      "keywords": [],
      "method_name_shortname": ""
    }}
    ```
    Choose field_of_Paper from exactly one of:
    ["Mathematics","Physics","Chemistry","Computer Science","Electrical Engineering",
    "Engineering","Materials Science","Astronomy","Earth Science","Biology",
    "Medicine","Economics","Political Science","Sociology","Psychology",
    "Linguistics","Philosophy","History","Geography","Arts"]
    
    Title: "{title}"
    Abstract: "{abstract}"
    """
    return prompt



def query_transformers_for_summaries(n_papers=10, model_name="Qwen/Qwen2.5-7B-Instruct", prompt_fn=prompt_with_text,
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
    paper_data_tuples = load_texts(n_papers)
    # check if output file already exists
    output_dict = {}
    if os.path.exists(output_path):
        print(f"Output file already exists at {output_path}. Continuing from where it left off.")
        paper_summaries = read_dict(output_path)
        processed_paper_ids = set(paper_summaries.keys())
        paper_data_tuples = [(paper_id, title, abstract) for paper_id, title, abstract in paper_data_tuples if paper_id not in processed_paper_ids]
        output_dict = paper_summaries

    # Initialize the model and tokenizer, ensuring model is on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"  # removed device_map to avoid offloading
    )
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, (paper_id, title, abstract) in enumerate(paper_data_tuples):
        if i % 10 == 0:
            print(f"Processing paper {i + 1} of {len(paper_data_tuples)}. Percent complete: {100 * (i + 1) / len(paper_data_tuples):.2f}%")
        prompt = prompt_fn(title, abstract)

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

        print(f"Paper ID: {paper_id}\nTitle: {title}\n\n{response}\nTime taken: {toc - tic:.2f} seconds\n")

        if (i + 1) % checkpoint_freq == 0:
            write_dict_to_pkl(output_dict, output_path)
    write_dict_to_pkl(output_dict, output_path)

    # Optionally free model and tokenizer memory at the end of the function
    del model, tokenizer
    torch.cuda.empty_cache()

    return output_dict


import os
import time
import torch


def query_qwen3_for_summaries(
    n_papers: int = 10,
    # model_name: str = "Qwen/Qwen3-8B",
    model_name: str = "Qwen/Qwen3-1.7B",
    prompt_fn = prompt_with_text,
    output_path: str = "../data/llm_outputs/llm_summaries_qwen3.pkl",
    checkpoint_freq: int = 10,
    model_params: dict = {
        "max_new_tokens": 512,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
        "do_sample": False,
    }
):
    """
    Query a Qwen 3 transformer model (with thinking‐mode) to generate summaries (labels + keywords)
    for a given set of papers. Returns a dict mapping paper_id to:
      {
        "thinking": <chain‐of‐thought trace>,
        "content":  <final summary>
      }
    """
    create_dirs_if_not_exist()
    paper_data = load_texts(n_papers)

    # resume if file exists
    if os.path.exists(output_path):
        print(f"Resuming from {output_path}")
        done = read_dict(output_path)
        seen = set(done.keys())
        paper_data = [(pid, t, a) for pid, t, a in paper_data if pid not in seen]
        output = done
    else:
        output = {}

    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"     # automatic sharding over GPUs/CPU
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # special token id for </think>
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")

    for i, (paper_id, title, abstract) in enumerate(paper_data):
        if i % checkpoint_freq == 0:
            print(f"Processing {i+1}/{len(paper_data)} ({100*(i+1)/len(paper_data):.1f}%)")
        # build prompt and apply chat template
        prompt = prompt_fn(title, abstract)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        # generate
        tic = time.time()
        with torch.no_grad():
            out_ids = model.generate(**inputs, **model_params)[0]
        toc = time.time()

        # strip off the input tokens
        gen_ids = out_ids[len(inputs.input_ids[0]):].tolist()

        # split thinking vs content
        try:
            # find the last </think>
            idx = len(gen_ids) - gen_ids[::-1].index(end_think_id)
        except ValueError:
            idx = 0

        think_ids   = gen_ids[:idx]
        content_ids = gen_ids[idx:]

        thinking = tokenizer.decode(think_ids,   skip_special_tokens=True).strip()
        content  = tokenizer.decode(content_ids, skip_special_tokens=True).strip()

        # store both
        # output[paper_id] = {
        #     "thinking": thinking,
        #     "content": content
        # }
        output[paper_id] = content

        print(f"Paper {paper_id} done in {toc-tic:.2f}s\nTHINKING:\n{thinking}\n\nFINAL:\n{content}\n")

        # checkpoint
        if (i+1) % checkpoint_freq == 0:
            write_dict_to_pkl(output, output_path)

    # final save & cleanup
    write_dict_to_pkl(output, output_path)
    del model, tokenizer
    torch.cuda.empty_cache()

    return output



def compare_two_runs(n_papers,
                     model_params1, model_params2,
                     prompt_fn1, prompt_fn2,
                     output_path1, output_path2,
                     model_name1="Qwen/Qwen2.5-7B-Instruct",
                     model_name2="Qwen/Qwen3-1.7B",
                     paper_ids_text_pairs_path="../data/llm_inputs/paper_ids_text_pairs.pkl",
                     checkpoint_freq=10,
                     run_inference1=True, run_inference2=True):
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
    if run_inference1:
        # Run 1: Using prompt_fn1 and model_params1
        print("Starting Run 1...")
        # clean output file if it exists
        if os.path.exists(output_path1):
            os.remove(output_path1)
        start_time_run1 = time.time()
        outputs1 = query_transformers_for_summaries(
            n_papers=n_papers,
            model_name=model_name1,
            prompt_fn=prompt_fn1,
            output_path=output_path1,
            checkpoint_freq=checkpoint_freq,
            model_params=model_params1
        )
        end_time_run1 = time.time()

        gc.collect()
        torch.cuda.empty_cache()
        print(f"Run 1 completed in {end_time_run1 - start_time_run1:.2f} seconds")
        print("Cleared GPU resources after Run 1.")

    if run_inference2:
        # Run 2: Using prompt_fn2 and model_params2
        print("\nStarting Run 2...")
        if os.path.exists(output_path2):
            os.remove(output_path2)
        start_time_run2 = time.time()
        outputs2 = query_qwen3_for_summaries(
            n_papers=n_papers,
            model_name=model_name2,
            prompt_fn=prompt_fn2,
            output_path=output_path2,
            checkpoint_freq=checkpoint_freq,
            model_params=model_params2
        )
        end_time_run2 = time.time()
        print(f"Run 2 completed in {end_time_run2 - start_time_run2:.2f} seconds")

    paper_triplets = load_texts(n_papers)
    abstracts_dict = {paper_id: {'title': title, 'abstract': abstract} for paper_id, title, abstract in paper_triplets}

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
                    print(f"Title: {abstracts_dict.get(paper_id, {}).get('title', '')}")
                    print(f"Abstract: {abstracts_dict.get(paper_id, {}).get('abstract', '')}")
                    diff_found = True
                print(f"{key}:\nRun 1: {result1.get(key)} \nRun 2: {result2.get(key)}")
        if diff_found:
            print("---------------------\n")
    print("\nSummary of differences:")
    for key, value in differences.items():
        print(f"{key}: {value} papers with different values.")
    print("\n")



def run_comparison(n_papers=10, run_inference1=True, run_inference2=True):

    model_params1 = {
        "max_new_tokens": 512,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.1,
        "do_sample": False}
    model_params2 = {
        "max_new_tokens": 512,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.1,
        "do_sample": False}

    prompt_fn1 = new_prompt_with_text_v3_0
    prompt_fn2 = new_prompt_with_text_v4_2

    output_path1 = "../data/llm_outputs/llm_summaries_test1.pkl"
    output_path2 = "../data/llm_outputs/llm_summaries_test2.pkl"

    # if delete_existing_outputs:
    #     for output_path in [output_path1, output_path2]:
    #         if os.path.exists(output_path):
    #             os.remove(output_path)

    compare_two_runs(n_papers, model_params1, model_params2, prompt_fn1, prompt_fn2, output_path1, output_path2,
                     run_inference1=run_inference1, run_inference2=run_inference2)




if __name__ == '__main__':
    # summaries = query_transformers_for_summaries(n_papers=1000000, model_name="Qwen/Qwen2.5-7B-Instruct",
    #                                              prompt_fn=new_prompt_with_text,
    #                                              output_path="../data/llm_outputs/llm_summaries_pretty20_fields.pkl",
    #                                              model_params={
    #                                                  "max_new_tokens": 512,
    #                                                  "temperature": None,
    #                                                  "top_k": None,
    #                                                  "top_p": None,
    #                                                  "repetition_penalty": 1.1,
    #                                                  "do_sample": False},
    #                                              checkpoint_freq=10)

    run_comparison(n_papers=10, run_inference1=False, run_inference2=True)
    # print(new_prompt_with_text_v4_0(1, 2))

    # load and parse the summaries
    # parse_and_save_llm_outputs("../data/llm_outputs/llm_summaries_transformers.pkl", "../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # loaded_summaries = read_dict_from_pkl("../data/llm_outputs/llm_summaries_transformers_parsed.pkl")
    # for paper_id, summary in loaded_summaries.items():
    #     print(f"Paper ID: {paper_id}")
    #     print(f"Summary: {summary}")
    #     print("\n")


