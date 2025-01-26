import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = []
input_message = input("Give a prompt:")
messages.append({"role":"user","content":input_message})
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
ans_text = outputs[0]["generated_text"]
ans_text = ans_text.replace("<|user|>\n"+input_message+"</s>\n<|assistant|>\n","")

print(ans_text)