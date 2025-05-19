import json
with open("llm_router_result_ranked_f.json", 'r', encoding='utf-8') as f:
    res = json.load(f)
# 对每一项，把列表换成字典，按顺序赋值
r = {}
for d in res:
    r[d] = {}
    len_d = len(res[d])
    for k in res[d]:
        r[d][k] = len_d
        len_d -= 1
    # 对没有的值 补充为分数0
    for k in ['nq', 'triviaqa', 'ott', 'tat', 'kg']:
        if k not in res[d]:
            r[d][k] = 0
# baocun
with open("llm_router_result_ranked_ff.json", 'w', encoding='utf-8') as f:
    json.dump(r, f, ensure_ascii=False, indent=4)