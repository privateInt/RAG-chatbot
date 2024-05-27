import torch
import torch.nn as nn
import transformers
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
from typing import Tuple
from tqdm import tqdm
from copy import deepcopy

from DPR_data import DPRDataset, DPRSampler, dpr_collator
from DPR_model import KobertBiEncoder
from DPR_make_passage_vector import *

class Trainer:
    def __init__(
        self,
        model,
        device,
        train_dataset,
        valid_dataset,
        num_epoch: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float],
        num_warmup_steps: int,
        num_training_steps: int,
        valid_every: int,
        best_val_ckpt_path: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=DPRSampler(
                train_dataset, batch_size=batch_size, drop_last=False
            ),
            collate_fn=lambda x: dpr_collator(
                x, padding_value=ds.pad_token_id
            ),
            num_workers=4,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_sampler=DPRSampler(
                valid_dataset, batch_size=batch_size, drop_last=False
            ),
            collate_fn=lambda x: dpr_collator(
                x, padding_value=ds.pad_token_id
            ),
            num_workers=4,
        )

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_every = valid_every
        self.lr = lr 
        self.betas = betas 
        self.num_warmup_steps = num_warmup_steps 
        self.num_training_steps = num_training_steps
        self.best_val_ckpt_path = best_val_ckpt_path

        self.start_ep = 1
        self.start_step = 1
        
    def calc_loss_acc(self, batch):
        _, q, q_mask, _, p, p_mask = batch # q_id, q, q_mask, p_id, p, p_mask
        q, q_mask, p, p_mask = (
            q.to(self.device),
            q_mask.to(self.device),
            p.to(self.device),
            p_mask.to(self.device),
        )
        
        q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
        p_emb = self.model(p, p_mask, "passage")  # bsz x bert_dim
        pred = torch.matmul(q_emb, p_emb.T)  # bsz x bsz
        
        bsz = pred.size(0)
        target = torch.arange(bsz)
        
        loss = torch.nn.functional.cross_entropy(pred, target.to(device))
        acc = (pred.detach().cpu().max(1).indices == target).sum().float() / bsz
        
        return loss, acc, bsz
    
    def fit(self):
        global_step_cnt = 0
        prev_best = None
        writer = SummaryWriter() # init tensorboard
        
        # if exist already model, load training_state
        if os.path.exists(self.best_val_ckpt_path):
            training_state = torch.load(self.best_val_ckpt_path)
            self.model.load_state_dict(training_state["model_state_dict"])
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            self.scheduler.load_state_dict(training_state["scheduler_state"])
            self.start_ep = training_state["epoch"]
            self.start_step = training_state["step"]
        
        for ep in range(self.start_ep, self.num_epoch + 1):
            for step, batch in enumerate(
                tqdm(self.train_loader, desc=f"epoch {ep} batch"), 1
            ):
                
                # train step
                if ep == self.start_ep and step < self.start_step:
                    continue  # 중간부터 학습시키는 경우 해당 지점까지 복원
                    
                self.model.train()
                global_step_cnt += 1
                train_loss, train_acc, _ = self.calc_loss_acc(batch)
                
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                training_state = {
                    "epoch": ep,
                    "step": step,
                    "global_step": global_step_cnt,
                    "train_step_loss": train_loss.cpu().item(),
                    "current_lr": float(self.scheduler.get_last_lr()[0]),
                    "train_step_acc": train_acc,
                    "optimizer_state": deepcopy(self.optimizer.state_dict()),
                    "scheduler_state": deepcopy(self.scheduler.state_dict()),
                    "model_state_dict": deepcopy(self.model.state_dict())
                }
                
                # valid step
                if global_step_cnt % self.valid_every == 0:
                    self.model.eval()
                    loss_list = []
                    sample_cnt = 0
                    tmp_valid_acc = 0
                    with torch.no_grad():
                        for batch in self.valid_loader:
                            loss, step_acc, batch_size = self.calc_loss_acc(batch)

                            sample_cnt += batch_size
                            tmp_valid_acc += step_acc * batch_size
                            loss_list.append(loss.cpu().item() * batch_size)
                            
                    valid_loss = np.array(loss_list).sum() / float(sample_cnt)
                    valid_acc = tmp_valid_acc / float(sample_cnt)

                    eval_dict = {
                        "valid_loss" : valid_loss,
                        "valid_acc" : valid_acc,
                    }
                    training_state.update(eval_dict)
                    
                    # valid step마다 학습 log저장
                    with open("train_log.log", "a")as f:
                        f.write(f"ep: {ep}\n")
                        f.write(f"step: {step}\n")
                        f.write(f"current_lr: {float(self.scheduler.get_last_lr()[0])}\n")
                        f.write(f"batch_size: {batch_size}\n")
                        f.write(f"global_step: {global_step_cnt}\n")
                        f.write(f"train_loss: {train_loss.cpu().item()}\n")
                        f.write(f"train_acc: {train_acc}\n")
                        f.write(f"valid_loss: {valid_loss}\n")
                        f.write(f"valid_acc: {valid_acc}\n\n\n")
                    
                    # valid loss가 가장 낮을때만 저장
                    if prev_best is None or eval_dict["valid_loss"] < prev_best:
                        prev_best = eval_dict["valid_loss"]
                        torch.save(training_state, self.best_val_ckpt_path)
                   
            # tensorboard log 저장
            writer.add_scalar("train_loss", train_loss, ep)
            writer.add_scalar("train_acc", train_acc, ep)
            writer.add_scalar("valid_loss", valid_loss, ep)
            writer.add_scalar("valid_acc", valid_acc, ep)
            writer.add_scalar("lr", float(self.scheduler.get_last_lr()[0]), ep)
                
        # tensorboard 종료              
        writer.flush()
        writer.close()
                        
if __name__ == "__main__":
    device = "cuda"
    dataset_name = "kullm_custom_data_240228.xlsx"
    
    ds = DPRDataset(f"../data/{dataset_name}", 0.2)
    train_dataset = ds.tokenized_data_tuples_train
    valid_dataset = ds.tokenized_data_tuples_test
    model = KobertBiEncoder()

    my_trainer = Trainer(
        model=model,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epoch=20, # default = 20
        batch_size=16, # default = 96
        lr=3e-4,
        betas=(0.9, 0.99),
        num_warmup_steps=1000,
        num_training_steps=100000,
        valid_every=30,
        best_val_ckpt_path="my_model.pt",
    )

    my_trainer.fit()
    
    # make_passage_vector
    
    dataset = PassageChunkDataset(
        target_retriever_path = f"../data/{dataset_name}", 
        chunk_size = 100, 
        batch_size = 64
    )

    save_passage_idx_pair_in_vector_db(
        target_retriever_path = f"../data/{dataset_name}", 
        encoder = KobertBiEncoder(),
        best_val_ckpt_path = "my_model.pt",
        device = "cuda",
        data_loader = dataset.create_dataloader()
    )