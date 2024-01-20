import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device_name = torch.cuda.get_device_name(0)  
print("Accelerator Device:", device_name)


model_name_or_path = "TheBloke/EverythingLM-13b-V2-16K-GPTQ"


model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            device_map="auto",
                                            trust_remote_code=True,  
                                            revision="main")


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


prompt = "explain about law of attraction in 50 words"


print("\n\n*** Generate:")

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
generated_text = tokenizer.decode(output[0])
print(generated_text)


with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)