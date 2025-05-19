import os
import json
from zhipuai import ZhipuAI
from openai import OpenAI
import time


class GLM:

    client = ZhipuAI(api_key='your api key')
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

# # OpenAi格式的GLM
# class GLM:
#     api_key = "'your api key'"
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
    api_key = "'your api key'"
    base_url = "base url"
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
        self.token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0

        api_key = ["your api key", "", "", "", "", ""]
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
    api_key = 'your api key'
    base_url = "based url"
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


if __name__ == '__main__':
    # glm_model = GLM('glm-4-plus')
    # print(glm_model.generate("你是什么模型"))


    gpt = GPT("gpt-4o")
    cnt = 0
    flag = True
    while flag:
        try:
            print(gpt.generate(STR))
            flag = False
        except:
            cnt += 1
    print(cnt)

    # deepseek = DeepSeek()
    # print(deepseek.generate("你是什么模型？"))
    #
    # claude = Claude()
    # print(claude.generate(STR))



