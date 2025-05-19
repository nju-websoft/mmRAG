import os
import json
from zhipuai import ZhipuAI
from openai import OpenAI
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GLM:
    # client = ZhipuAI(api_key='5316a71460a52b0f0867ffb16f8484de.q8yQcRIho55cHamd')
    client = ZhipuAI(api_key='9e5f4303357912d65f47aeb9cd1165ce.5WQ6BsThAelM4mAf')
    def __init__(self, model_name):
        self.model_name = model_name
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

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
        self.input_token_count += response.usage.prompt_tokens
        self.output_token_count += response.usage.completion_tokens
        # print("token used: ", response.usage.total_tokens)
        # print("input token used: ", response.usage.prompt_tokens)
        # print("output token used: ", response.usage.completion_tokens)
        return response.choices[0].message.content

    def get_token_count(self):
        return self.token_count, self.input_token_count, self.output_token_count

    def reset_token_count(self):
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

# # OpenAi格式的GLM
# class GLM:
#     api_key = "sk-b9fd118f7c1c4da0997c1bfb6c990742"
#     base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#     client = OpenAI(api_key=api_key, base_url=base_url)
#     def __init__(self, model_name="gpt-4o"):
#         self.model_name = model_name
#         self.model_name = "qwen-plus"
#         self.token_count = 0
#         self.input_token_count = 0
#         self.output_token_count = 0
#
#     def __str__(self):
#         return "gpt"
#
#     def add_token(self, token):
#         self.token_count += token
#
#     def generate(self, messages):
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": messages,
#                 }
#             ]
#         )
#
#         self.add_token(response.usage.total_tokens)
#         self.input_token_count += response.usage.prompt_tokens
#         self.output_token_count += response.usage.completion_tokens
#         print("token used: ", response.usage.total_tokens)
#         print("input token used: ", response.usage.prompt_tokens)
#         print("output token used: ", response.usage.completion_tokens)
#         return response.choices[0].message.content
#
#     def get_token_count(self):
#         return self.token_count, self.input_token_count, self.output_token_count
#
#     def reset_token_count(self):
#         self.token_count = 0
#         self.input_token_count = 0
#         self.output_token_count = 0


class GPT:
    # api_key = "sk-6lPeSn4rxWYoEtPl877aC20eCc304c4388C1131289F21929"
    # api_key = "sk-Ltivm0CyLMmnb3RTA56a1b21A6964d30B6F14574DfBfF395"
    api_key = "sk-RGJrVjruSf8zIiaDa7g63FC9TnpUYTWgGMQrVPGvh5KuKYpW"
    # base_url = "https://api.yesapikey.com/v1"
    base_url = "https://api.chatanywhere.tech/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    def test(self):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    # max_tokens=10000,
                    # temperature=1,
                    # top_p=1,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ]
                )
                # 判断有没有补全内容
                if getattr(completion.choices[0].message, 'content', None):
                    content = completion.choices[0].message.content
                    print(completion)  # 完整返回值
                    print(content)  # 提取补全内容
                    break
                else:
                    # 如果没有内容，打印错误信息或提示
                    # print(completion)
                    print('error_wait_2s')
            except Exception as e:
                print(e)
                print('error_wait_2s1')
            time.sleep(2)

    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def __str__(self):
        return "gpt"

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
        self.input_token_count += response.usage.prompt_tokens
        self.output_token_count += response.usage.completion_tokens
        print("token used: ", response.usage.total_tokens)
        print("input token used: ", response.usage.prompt_tokens)
        print("output token used: ", response.usage.completion_tokens)
        return response.choices[0].message.content

    def get_token_count(self):
        return self.token_count, self.input_token_count, self.output_token_count

    def reset_token_count(self):
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0


class DeepSeek:
    def __init__(self, model_name="deepseek-chat", par_index=0):
        self.model_name = model_name
        # self.model_name = "deepseek-ai/DeepSeek-V3"
        # self.model_name = "claude-3-5-sonnet-20240620"
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

        api_key = ["sk-82c3cb3b8ba44427990b1c8769984b4e", "sk-52ca173ffcbb4cf6af6291bd2f22785e", "sk-e1db8127f1034419b54b5aee0aead2cc", "sk-57914c5305f54638acc98f12162c4dfe", "sk-c4b566b4eb6d477ba2cacbd0818a8b35", "sk-9a97b07d59de44e8b5358b7ab4f475bd"]
        # api_key = ["sk-xwqklzccicsrubpulftskpptdbfdcbnvbolwplovriishmbl"]
        # api_key = ["sk-jEHCMJ9cbRq5KgGQB82aB265FfF643Cf9736EcD0F0B965F7"]  # claude
        base_url = "https://api.deepseek.com"
        # base_url = "https://api.siliconflow.cn/v1"
        # base_url = "https://api.yesapikey.com/v1"  # claude
        self.client = OpenAI(api_key=api_key[par_index], base_url=base_url)

    def __str__(self):
        return "deepseek"

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
            ],
            temperature = 0
        )

        self.add_token(response.usage.total_tokens)
        self.input_token_count += response.usage.prompt_tokens
        self.output_token_count += response.usage.completion_tokens
        print("token used: ", response.usage.total_tokens)
        print("input token used: ", response.usage.prompt_tokens)
        print("output token used: ", response.usage.completion_tokens)
        return response.choices[0].message.content

    def get_token_count(self):
        return self.token_count, self.input_token_count, self.output_token_count

    def reset_token_count(self):
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0


class Claude:
    api_key = "sk-jEHCMJ9cbRq5KgGQB82aB265FfF643Cf9736EcD0F0B965F7"
    base_url = "https://api.yesapikey.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    def test(self):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="claude-3-5-sonnet-20240620",
                    # max_tokens=10000,
                    # temperature=1,
                    # top_p=1,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ]
                )
                # 判断有没有补全内容
                if getattr(completion.choices[0].message, 'content', None):
                    content = completion.choices[0].message.content
                    print(completion)  # 完整返回值
                    print(content)  # 提取补全内容
                    break
                else:
                    # 如果没有内容，打印错误信息或提示
                    # print(completion)
                    print('error_wait_2s')
            except Exception as e:
                print(e)
                print('error_wait_2s1')
            time.sleep(2)

    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        self.model_name = model_name
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

    def __str__(self):
        return "claude"

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
        self.input_token_count += response.usage.prompt_tokens
        self.output_token_count += response.usage.completion_tokens
        print("token used: ", response.usage.total_tokens)
        print("input token used: ", response.usage.prompt_tokens)
        print("output token used: ", response.usage.completion_tokens)
        return response.choices[0].message.content

    def get_token_count(self):
        return self.token_count, self.input_token_count, self.output_token_count

    def reset_token_count(self):
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0


class Qwen2p5:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map={"": "cuda:0"},
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.model.eval()

    def generate(
            self,
            input: str,
            max_new_tokens: int = 32,
            temperature: float = 0.1,
            top_p: float = 0.9,
            **gen_kwargs
    ) -> str:
        messages = [
            {"role": "system", "content": "You shuold follow my instructino and DO NOT OUTPUT ANY UNRELATED WORDS"},
            {"role": "user", "content": input}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **gen_kwargs
        )
        gen_ids = output_ids[0, input_ids.shape[-1]:]
        ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(ans)
        return ans


if __name__ == '__main__':
    # glm_model = GLM('glm-4-plus')
    # print(glm_model.generate("你是什么模型"))

    # import pickle
    # with open("error.pkl", "rb") as f:
    #     STR = pickle.load(f)
    # with open("error_input.txt", "w") as f:
    #     f.write(STR)
    #
    # print(STR)
    #
    #
    # gpt = GPT("gpt-4o")
    # cnt = 0
    # flag = True
    # while flag:
    #     try:
    #         print(gpt.generate(STR))
    #         flag = False
    #     except:
    #         cnt += 1
    # print(cnt)

    qwen = Qwen2p5("Qwen/Qwen2.5-7B-Instruct")
    print(qwen.generate("你是什么模型？你是QWEN吗？"))

    # deepseek = DeepSeek()
    # print(deepseek.generate("你是什么模型？"))
    #
    # claude = Claude()
    # print(claude.generate(STR))



