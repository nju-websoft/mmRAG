from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from zhipuai import ZhipuAI


class Qwen:
    cache_dir = ""

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, local_files_only=True)
        print("Model loaded")

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def complete(self, prompt):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.1
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


class GLM:
    client = ZhipuAI(api_key='your api key')
    def __init__(self, model_name):
        self.model_name = model_name
        self.token_count = 0

    def __str__(self):
        return "glm"

    def add_token(self, token):
        self.token_count += token

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": messages,
                }
            ]
        )

        self.add_token(response.usage.total_tokens)
        return response.choices[0].message.content

    def get_token_count(self):
        return self.token_count

    def reset_token_count(self):
        self.token_count = 0
