from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "../data/LLMs/qwen2.5-coder-7b-instruct-q6_k.gguf"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = '''Adaptive DCTNet for Audio Signal Classification. In this paper, we investigate DCTNet for audio signal classification. Its
                output feature is related to Cohen's class of time-frequency distributions. We
                introduce the use of adaptive DCTNet (A-DCTNet) for audio signals feature
                extraction. The A-DCTNet applies the idea of constant-Q transform, with its
                center frequencies of filterbanks geometrically spaced. The A-DCTNet is
                adaptive to different acoustic scales, and it can better capture low frequency
                acoustic information that is sensitive to human audio perception than features
                such as Mel-frequency spectral coefficients (MFSC). We use features extracted
                by the A-DCTNet as input for classifiers. Experimental results show that the
                A-DCTNet and Recurrent Neural Networks (RNN) achieve state-of-the-art
                performance in bird song classification rate, and improve artist identification
                accuracy in music data. They demonstrate A-DCTNet's applicability to signal
                processing problems.'''

    prompt = f"""Given the abstract below, please extract and label the following key details clearly. Each response should be straightforward and specific:
                field_of_Paper = * The primary academic discipline * [insert answer];
                subfield = * The main research category within the field * [insert answer];
                sub_subfield = * A narrower focus within the subfield * [insert answer];
                keywords = * A set of 3-5 words or phrases that describe the core topics. * [insert answer];
                method_name_shortname = *The main technique or model name proposed in the abstract.* [insert answer];
                
                Abstract: '{text}'
                
                Ensure each label is followed by a concise answer with minimal formatting issues, 
                and separate multiple keywords with commas. 
                If information is not clear from the abstract, write "N/A" for that field."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


