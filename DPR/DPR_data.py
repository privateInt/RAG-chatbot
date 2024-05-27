import os
import pickle
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import BertTokenizerFast
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
from typing import Iterator, List, Sized, Tuple

def dpr_collator(batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
    """q_id, query, p_id, gold_passage를 batch로 반환합니다."""
    batch_q_id = T([e[0] for e in batch])[:, None]
    batch_q = pad_sequence(
        [T(e[1]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = T([e[2] for e in batch])[:, None]
    batch_p = pad_sequence(
        [T(e[3]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_q_id, batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask)


class DPRSampler(torch.utils.data.BatchSampler):
    """in-batch negative학습을 위해 batch 내에 중복 answer를 갖지 않도록 batch를 구성합니다.
    sample 일부를 pass하기 때문에 전체 data 수보다 iteration을 통해 나오는 데이터 수가 몇십개 정도 적습니다."""

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        generator=None,
    ) -> None:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(
                data_source, replacement=False, generator=generator
            )
        else:
            sampler = torch.utils.data.SequentialSampler(data_source)
        super(DPRSampler, self).__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            if item[1] in sampled_p_id:
                continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
            sampled_idx.append(idx)
            sampled_p_id.append(item[1])
            if len(sampled_idx) >= self.batch_size:
                yield sampled_idx
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last:
            yield sampled_idx

class DPRDataset:
    def __init__(self, excel_path: str, split_ratio: float):
        self.excel_path = excel_path
        self.split_ratio = split_ratio
        self.preprocessed_excel_path_train = excel_path.replace(os.path.basename(excel_path), os.path.splitext(excel_path)[0] + f"_train_{1-self.split_ratio}.p")
        self.preprocessed_excel_path_test = excel_path.replace(os.path.basename(excel_path), os.path.splitext(excel_path)[0] + f"_test_{self.split_ratio}.p")
        self.matching_test_file = excel_path.replace(os.path.basename(excel_path), os.path.splitext(excel_path)[0] + f"_for_retriever_src.p")
        
        
        self.pivot_key = "제시문"
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        
        self.tokenized_data_tuples_train = []
        self.tokenized_data_tuples_test = []
        self.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]
        self.data_load()
        
    def excel_preprocessing(self):
        df = pd.read_excel(self.excel_path)
        
        tmp_result = defaultdict(list)
        tmp_lst = []
        for idx, i in enumerate(tqdm(df.index)):
            save_target_dict = {}
            tmp_dict = {}
            tmp_input_dict = {}
            for column_name in df.columns: # data type check
                tmp_dict[column_name] = df[column_name][i] if isinstance(df[column_name][i], str) else ''
                
            # {pivot_key_context: [{q:q_context, a:a_context, r: r_context, q_id: idx}]}
            tmp_input_dict = {column: tmp_dict[column] for column in df.columns if column != self.pivot_key}
            tmp_input_dict["q_id"] = idx
            save_target_dict[tmp_dict[self.pivot_key]] = tmp_input_dict
            tmp_lst.append(save_target_dict)
            
        # collect with same key
        # {pivot_key_context: [{q:q_context, a:a_context, r: r_context, q_id: idx}, {q:q_context, a:a_context, r: r_context, q_id: idx}, ...]}
        for data in tmp_lst:
            key = list(data.keys())[0]
            value = list(data.values())[0]
            tmp_result[key].append(value)
            
        # assign index to pivot_key_context
        # {pivot_key_context: {'0': [{q:q_context, a:a_context, r: r_context, q_id: idx}, {q:q_context, a:a_context, r: r_context, q_id: idx}, ...]}}
        final_result = {}
        for idx, pivot_key_context in enumerate(tmp_result):
            final_result[pivot_key_context] = {idx: tmp_result[pivot_key_context]}
            
        return final_result
        
    def train_test_split(self, input_dict):
        keys = list(input_dict.keys())
        values = list(input_dict.values())
        
        keys_train, keys_test, values_train, values_test = train_test_split(keys, values, test_size = self.split_ratio, random_state = 42)
        
        input_dict_train = {k: input_dict[k] for k in keys_train}
        input_dict_test = {k: input_dict[k] for k in keys_test}
        
        return input_dict_train, input_dict_test
            
    def extract_question_passageIDX_passage(self, input_dict):
        tmp_lst = []
        for passage, inner_data in tqdm(input_dict.items()):
            for passage_id, sub_data in inner_data.items():
                for i in sub_data:
                    tmp_lst.append((i["q_id"], i["질문"], passage_id, passage))
                    
        return tmp_lst
                
    def convert_data_to_token(self, input_list):
        """
        tokenizer의 최대 토큰 수까지 token_id를 추출하기 위해
        truncation=True, max_length = None 옵션을 사용하고
        token_id만 쓰기 위해 "input_ids"만 추출
        """
        tokenized_data_tuples = [
            (query_id, self.tokenizer(query, truncation=True, max_length = None)["input_ids"], passage_id, self.tokenizer(passage, truncation=True, max_length = None)["input_ids"]) 
            for query_id, query, passage_id, passage in tqdm(input_list)
        ]
        
        return tokenized_data_tuples
    
    def data_load(self):
        # 원본 파일 저장
        with open(self.matching_test_file, "wb")as f:
            pickle.dump(self.excel_preprocessing(), f)
            
        # 이미 생성된 파일이 있는 경우 읽어오기
        if os.path.exists(self.preprocessed_excel_path_train) and os.path.exists(self.preprocessed_excel_path_test):
            with open(self.preprocessed_excel_path_train, "rb") as f:
                self.tokenized_data_tuples_train = pickle.load(f)
            with open(self.preprocessed_excel_path_test, "rb") as f:
                self.tokenized_data_tuples_test = pickle.load(f) 
                
        # 이미 생성된 파일이 없는 경우 새로 생성
        else:
            tmp_train, tmp_test = self.train_test_split(self.excel_preprocessing())
            self.tokenized_data_tuples_train = self.convert_data_to_token(self.extract_question_passageIDX_passage(tmp_train))
            self.tokenized_data_tuples_test = self.convert_data_to_token(self.extract_question_passageIDX_passage(tmp_test))
            
            with open(self.preprocessed_excel_path_train, "wb") as f:
                pickle.dump(self.tokenized_data_tuples_train, f)
            with open(self.preprocessed_excel_path_test, "wb") as f:
                pickle.dump(self.tokenized_data_tuples_test, f)
                
if __name__ == "__main__":
    ds = DPRDataset(excel_path = "../data/kullm_custom_data_240228.xlsx", split_ratio = 0.2)
    train_dataset = ds.tokenized_data_tuples_train
    test_dataset = ds.tokenized_data_tuples_test
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=DPRSampler(train_dataset, batch_size=16, drop_last=False),
        collate_fn=lambda x: dpr_collator(x, padding_value=ds.pad_token_id),
        num_workers=4,
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_sampler=DPRSampler(test_dataset, batch_size=16, drop_last=False),
        collate_fn=lambda x: dpr_collator(x, padding_value=ds.pad_token_id),
        num_workers=4,
    )
    
    train_cnt = 0
    for batch in tqdm(train_loader):
        train_cnt += batch[0].size(0)
    # print(train_cnt)
    
    test_cnt = 0
    for batch in tqdm(test_loader):
        test_cnt += batch[0].size(0)
    # print(test_cnt)
    
    # print(train_cnt + test_cnt)
    # print(train_dataset)
    # print(test_dataset)
