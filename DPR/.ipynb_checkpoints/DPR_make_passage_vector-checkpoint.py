import torch
from torch import tensor as T
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast
import pickle
from typing import List, Tuple
import os
import numpy as np
import faiss
from DPR_model import KobertBiEncoder
import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()  # 토크나이저 초기화 관련 warning suppress

class PassageChunkDataset(IterableDataset):
    def __init__(self, target_retriever_path: str, chunk_size: int, batch_size: int):
        self.target_retriever_path = os.path.splitext(target_retriever_path)[0] + "_for_retriever_src.p"
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.passage_chunk_lst = []
        self.retriever_dict = {}
        
        self.load_passage_chunks() 
        
    def load_passage_chunks(self):
        chunk_idx = 0
        with open(self.target_retriever_path, "rb") as f:
            data_dict = pickle.load(f)
        for idx, passage in enumerate(data_dict):
            encoded_passage = self.tokenizer.encode(passage)
            root = data_dict[passage][idx][0]["출처"]
            for start_idx in range(0, len(encoded_passage), self.chunk_size):
                end_idx = min(len(encoded_passage), start_idx + self.chunk_size)
                chunk = encoded_passage[start_idx:end_idx]
                self.passage_chunk_lst.append((chunk, chunk_idx))
                self.retriever_dict[chunk_idx] = (passage, root)
                chunk_idx += 1
        with open(self.target_retriever_path.replace("_for_retriever_src.p", "_for_retriever_dst.p"), "wb")as f:
            pickle.dump(self.retriever_dict, f)

    def __iter__(self):
        for data in self.passage_chunk_lst:
            yield data

    def passage_collator(self, batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
        batch_p = pad_sequence(
            [torch.tensor(e[0]) for e in batch], batch_first=True, padding_value=padding_value
        )
        batch_p_attn_mask = (batch_p != padding_value).long()
        batch_p_id = batch_p_id = T([e[1] for e in batch])[:, None]
        
        return batch_p, batch_p_attn_mask, batch_p_id

    def create_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.passage_collator(
                x, padding_value=self.tokenizer.get_vocab()["[PAD]"]
            ),
            num_workers=1,
            worker_init_fn=None
        )

class save_passage_idx_pair_in_vector_db:
    def __init__(self, target_retriever_path, encoder, best_val_ckpt_path, device, data_loader):
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        self.root = os.path.splitext(target_retriever_path)[0]
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.load_state_dict(torch.load(best_val_ckpt_path)["model_state_dict"])
        
        self.data_loader = data_loader
        self.preprocessed_data = []
        
        self.index_path = self.root + "_passage_index.dpr"
        self.index_meta_path = self.root + "_passage_index_meta.dpr"
        self.index = faiss.IndexFlatIP(encoder.passage_encoder.pooler.dense.out_features) # for cosine similarity
        self.index_meta = []
        self.save_process()
    
    def save_process(self):
        cur = 0
        for batch in tqdm(self.data_loader, desc = "indexing"):
            p, p_mask, p_id = batch
            p, p_mask = p.to(self.device), p_mask.to(self.device)
            with torch.no_grad():
                p_emb = self.encoder(p, p_mask, "passage")
            self.preprocessed_data += [(idx, _emb) for idx, _emb in zip(p_id.cpu().numpy(), p_emb.cpu().numpy())]
            cur += p_emb.size(0)
        
        chunked_passage_idx = [t[0] for t in self.preprocessed_data]
        vectors = [np.reshape(t[1], (1, -1)) for t in self.preprocessed_data]
        vectors = np.concatenate(vectors, axis=0)
        self.index.add(vectors)
        self.index_meta.extend(chunked_passage_idx)
        
        faiss.write_index(self.index, self.index_path) # save index.dpr
        with open(self.index_meta_path, "wb") as f: # save index_meta.dpr
            pickle.dump(self.index_meta, f)
        
if __name__ == "__main__":
    dataset = PassageChunkDataset(
        target_retriever_path = "../data/kullm_custom_data_240228.xlsx", 
        chunk_size = 100, 
        batch_size = 64
    )

    save_passage_idx_pair_in_vector_db(
        target_retriever_path = "../data/kullm_custom_data_240228.xlsx", 
        encoder = KobertBiEncoder(),
        best_val_ckpt_path = "my_model.pt",
        device = "cuda",
        data_loader = dataset.create_dataloader()
    )