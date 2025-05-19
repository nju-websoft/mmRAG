import os
import json

folders = os.listdir("../final_result_marked")

# 保存数据集列表
mmrag_ds = []  # [{id, query, answer, relevant_chunks}, {}]

for folder in folders:
    if folder == ".DS_Store":
        continue
    if folder.endswith("_sample4"):
        continue
    print()
    print(folder)
    files = os.listdir("../final_result_marked/" + folder)
    for file in files:
        if file == ".DS_Store":
            continue
        with open("../final_result_marked/" + folder + "/" + file, "r") as f:
            data = json.load(f)
            qa = data['qa']
            llm_fail = qa['llm_fail']
            if len(llm_fail) > 0:
                continue
            query = qa['question']
            answer = qa['answer']
            in_ds_index = data['index']
            if folder in ['cwq', 'webqsp']:
                ori_context = data['context']
            else:
                ori_context = [f'{folder}_{in_ds_index}']
            relevant_context = {}
            keys = list(set(qa["llm_mark_result"].keys()))
            for key in keys:
                glm_result = -1
                ds_result = -1
                result = -1
                try:
                    glm_result = qa["llm_mark_result"][key]["Relevance Judgement"]
                    glm_result = int(glm_result)
                except:
                    continue
                try:
                    ds_result = qa["llm_mark_result_ds"][key]["Relevance Judgement"]
                    ds_result = int(ds_result)
                except:
                    continue
                if glm_result != ds_result:
                    try:
                        gpt_result = qa["llm_mark_result_gpt"][key]["Relevance Judgement"]
                        gpt_result = int(gpt_result)
                    except:
                        continue
                    if gpt_result == glm_result:
                        result = glm_result
                    elif gpt_result == ds_result:
                        result = ds_result
                    else:
                        result = 1
                else:
                    result = glm_result
                if result > 0:
                    relevant_context[key] = result
            save_dic = {'id': f'{folder}_{in_ds_index}', 'query': query, 'answer': answer, 'relevant_chunks': relevant_context, 'ori_context': ori_context}
            mmrag_ds.append(save_dic)
with open("mmrag_ds.json", "w") as f:
    json.dump(mmrag_ds, f, indent=4, ensure_ascii=False)
