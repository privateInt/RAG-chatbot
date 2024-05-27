import os
import pickle
from tqdm import tqdm
from transformers import BertTokenizerFast

from DPR_inference import inference
from DPR_model import KobertBiEncoder

target_data = "kullm_custom_data_240228.xlsx"

with open(os.path.splitext(target_data)[0] + "_test_0.2.p", "rb")as f:
    data = pickle.load(f)
    
tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

retriever = inference(
    target_retriever_path = target_data, 
    encoder = KobertBiEncoder(),
    best_val_ckpt_path = "my_model.pt",
    device = "cuda",
)

top_k_lst = [1,3]

for top_k in top_k_lst:
    cnt = 0
    for i in tqdm(data):
        q_id = i[0]
        q = i[1]
        p_id = i[2]
        p = i[3]
        
        query = tokenizer.decode(q, skip_special_tokens=True)
        label = tokenizer.decode(p, skip_special_tokens=True)

        result = retriever.retriever(query, top_k)
        
        result_lst = [passage for passage, similarity in result.items()]
        
        for i in result_lst:
            if label in i:
                cnt += 1
        
    print(f"top_k = {top_k}, acc = {cnt / len(data)}")