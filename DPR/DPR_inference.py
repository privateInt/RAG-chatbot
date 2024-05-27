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
import transformers
from tqdm import tqdm

from DPR_model import KobertBiEncoder
from DPR_make_passage_vector import PassageChunkDataset, save_passage_idx_pair_in_vector_db

class inference:
    def __init__(self, target_retriever_path, encoder, best_val_ckpt_path, device):
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        self.target_retriever_path = target_retriever_path
        self.root = os.path.splitext(target_retriever_path)[0]
        self.index_path = self.root + "_passage_index.dpr"
        self.index_meta_path = self.root + "_passage_index_meta.dpr"
        self.retriever_result_matching_path = self.root + "_for_retriever_dst.p"
        self.best_val_ckpt_path = best_val_ckpt_path
        
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.load_state_dict(torch.load(self.best_val_ckpt_path)["model_state_dict"])
        self.encoder.eval()
        
        self.index = faiss.IndexFlatIP(encoder.passage_encoder.pooler.dense.out_features) # for cosine similarity
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.index_path) and os.path.exists(self.index_meta_path) and os.path.exists(self.retriever_result_matching_path):
            self.index = faiss.read_index(self.index_path)
            self.index_meta = pickle.load(open(self.index_meta_path, "rb"))
            self.matching_dict = pickle.load(open(self.retriever_result_matching_path, "rb"))
        else:
            dataset = PassageChunkDataset(
                target_retriever_path = self.target_retriever_path, 
                chunk_size = 100, 
                batch_size = 64
            )

            save_passage_idx_pair_in_vector_db(
                target_retriever_path = self.target_retriever_path, 
                encoder = KobertBiEncoder(),
                best_val_ckpt_path = self.best_val_ckpt_path,
                device = self.device,
                data_loader = dataset.create_dataloader()
            )
            
            self.index = faiss.read_index(self.index_path)
            self.index_meta = pickle.load(open(self.index_meta_path, "rb"))
            self.matching_dict = pickle.load(open(self.retriever_result_matching_path, "rb"))
            
    def search_knn(
        self, query_vectors: np.array, top_docs: int
    ) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        chunked_passage_idx = [
            [self.index_meta[i] for i in query_top_idxs]
            for query_top_idxs in indexes
        ]
        result = [(chunked_passage_idx[i], scores[i]) for i in range(len(chunked_passage_idx))]
        return result
    
    def retriever(self, query, top_k):
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.encoder(T(tok["input_ids"]).to(self.device), T(tok["attention_mask"]).to(self.device), "query")
        final_result = self.search_knn(query_vectors = out.cpu().numpy(), top_docs = top_k)
        
        # 중복제거
        result_dict = {}
        for idx, sim in zip(*final_result[0]):
            result_dict[self.matching_dict[int(idx)][0]] = (sim, self.matching_dict[int(idx)][1])
        return result_dict
    
if __name__ == "__main__":
    retriever = inference(
        target_retriever_path = "../data/kullm_custom_data_240228.xlsx", 
        encoder = KobertBiEncoder(),
        best_val_ckpt_path = "my_model.pt",
        device = 'cuda'
    )

    result = retriever.retriever("비비탄총 수입하고 싶어", 20)
    print(result)
    for passage, tmp in result.items():
        print(f"passage: {passage}")
        print(f"similarity: {tmp[0]}")
        print(f"root: {tmp[1]}")