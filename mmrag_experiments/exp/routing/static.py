import json

with open("llm_router_result_ranked.json", 'r', encoding='utf-8') as f:
    router_res = json.load(f)
    new_res = {}
    # [No, NQ, TriviaQA. OTT, TAT, CWQ, WebQSP] -> ['No', 'nq', 'triviaqa', 'ott', 'tat', 'kg']
    for item in router_res:
        new = []
        for ds in router_res[item]:
            if ds == "No":
                new.append("No")
            elif ds in ['CWQ', 'WebQSP']:
                new.append("kg")
            elif ds in ['NQ', 'TriviaQA', 'OTT', 'TAT']:
                new.append(ds.lower())
        new = list(set(new))
        new_res[item] = new
with open("llm_router_result_ranked_f.json", 'w', encoding='utf-8') as f:
    json.dump(new_res, f, ensure_ascii=False, indent=4)