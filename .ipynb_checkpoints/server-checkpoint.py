'''
｢ 관세청 개인통관 안내 GPT  ｣

'''
# On-premise bot settings
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "data"))
sys.path.append(os.path.join(os.getcwd(), "DPR"))
sys.path.append(os.path.join(os.getcwd(), "LLM"))

from DPR.DPR_inference import inference
from DPR.DPR_model import KobertBiEncoder
from utils import Prompter

import torch, json, random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

# remove_text_lst = ['개인통관고유부호 확인은 관세청 개인통관고유부호 신청 및 조회 사이트(https://unipass. customs.go.kr)에서 공인인증서 또는 휴대전화로 본인인증 절차를 거친 후 가능합니다.', '세범위(US800)를 초과하는 것은 없으나 물품가격의 총합계가 US$ 1,100이므로 세관신고 대상']
# remove_root_lst = ['선물 및 국내 면세점 구매물품의 이중과세를 피하려면?']

class setup_engine:       
    def kullm(self):
        MODEL = "LLM/kullm_13b_custom_3072_6144_epochs20/"
        device_map = "auto"

        config = json.load(open(f"{MODEL}adapter_config.json"))

        trained_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained(config["base_model_name_or_path"])
        trained_model.eval()

        pipe = pipeline("text-generation", model=trained_model, tokenizer=tokenizer)
        prompter = Prompter("kullm")

        return pipe, prompter

    def retrieval(self):
        retriever = inference(
            target_retriever_path = "data/kullm_custom_data_240228.xlsx", 
            encoder = KobertBiEncoder(),
            best_val_ckpt_path = "DPR/my_model.pt",
            device = 'cuda'
        )

        return retriever    

def make_list(response):
    out_msg_lst = []
    if response:
        for p, tmp in response.items():
            passage = p
            sim = tmp[0]
            root = tmp[1]
            
#             for target in remove_root_lst:
#                 if target in passage:
#                     continue
            
#             for target in remove_text_lst:
#                 if target in root:
#                     continue
                    
            out_msg = f"[ 출처: {root} ]\n{passage}"
            out_msg_lst.append(out_msg)
    else:
        out_msg = "질문을 잘 이해하지 못 했습니다."
        out_msg_lst.append(out_msg)
            
    return out_msg_lst
    

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

setup = setup_engine() # class사용하려면 instance화 먼저 하기!

pipe, prompter = setup.kullm()
# retriever_traveler = setup.retrieval('traveler')
# retriever_commerce = setup.retrieval('commerce')
# retriever_moving = setup.retrieval('moving')
# retriever_post = setup.retrieval('post')
retriever_all = setup.retrieval()


@app.route('/kullm', methods=['POST'])
def kullm():
    data_dict = json.loads(request.data)
    
    prompt = data_dict['prompt']
    k = int(data_dict['top_k'])

    relevant_docs = retriever_all.retriever(query=prompt, top_k=k)
    relevant_doc_lst = make_list(relevant_docs)
    
#     tmp_lst = []
#     for i in relevant_doc_lst:
#         if not '의 이중과세를 피하려면?, page: 6' in i:
#             tmp_lst.append(i)
            
#     relevant_doc_lst = tmp_lst
#     print(relevant_doc_lst)
    
    # 현재 첫번째 참조문서만을 instruction으로 사용중
    answer = infer(instruction = relevant_doc_lst[0], input_text = prompt)
    
    result_dict = {
        'input_prompt': prompt,
        'doc_lst': relevant_doc_lst,
        'answer': answer
    }
    
    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    with open(f"{folder_name}/generation.log", "a")as f:
        f.write(str(datetime.now()) + "\n")
        f.write(f"사용자 입력값: {prompt}\n")
        f.write(f"출력 결과: {answer}\n")
        f.write(f"참조한 문서 목록: {relevant_doc_lst}\n")
        f.write("-"*30 + '\n')
        
    
    return json.dumps(result_dict, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8609)