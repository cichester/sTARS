import os
import logging
import torch
import torch.nn.functional as F
import polars as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW
from config import (
    TRIPLETS_PATH, ROBERTA_MODEL_NAME, ROBERTA_ADAPTER_DIR,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, FINETUNE_GRAD_ACCUM,
    FINETUNE_LR, FINETUNE_MAX_SEQ_LEN, TRIPLET_MARGIN, RANDOM_SEED
)
import numpy as np

logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    def __init__(self, df: pl.DataFrame, tokenizer, max_length: int):
        self.anchor = df["anchor_text"].to_list()
        self.positive = df["positive_text"].to_list()
        self.negative = df["negative_text"].to_list()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.anchor)

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        return {
            "anchor": self._tokenize(self.anchor[idx]),
            "positive": self._tokenize(self.positive[idx]),
            "negative": self._tokenize(self.negative[idx])
        }

def mean_pooling(model_output, attention_mask):
    """Mean pooling taking attention mask into account"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def train_roberta_lora():
    if os.path.exists(ROBERTA_ADAPTER_DIR):
        logger.info(f"Adapter già presente in {ROBERTA_ADAPTER_DIR}. Skip fine-tuning.")
        return

    logger.info("Avvio Fine-Tuning RoBERTa + LoRA su Triple Contrastive...")
    
    # 1. Caricamento Dataset
    if not os.path.exists(TRIPLETS_PATH):
        logger.error(f"Dataset triple {TRIPLETS_PATH} non trovato. Esegui il mining prima.")
        return
        
    df = pl.read_parquet(TRIPLETS_PATH)
    logger.info(f"Caricate {df.height} triple.")
    
    # Split train/val (90/10)
    df = df.sample(fraction=1.0, seed=RANDOM_SEED) # Shuffle
    split_idx = int(df.height * 0.9)
    train_df = df.slice(0, split_idx)
    val_df = df.slice(split_idx, df.height - split_idx)

    # 2. Modello & Tokenizer
    logger.info(f"Caricamento {ROBERTA_MODEL_NAME} con 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModel.from_pretrained(
        ROBERTA_MODEL_NAME, 
        quantization_config=bnb_config,
        device_map="auto",
        add_pooling_layer=False
    )
    
    # 3. Preparazione al training kbit e Applicazione LoRA
    logger.info("Configurazione PEFT/LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. DataLoader
    train_dataset = TripletDataset(train_df, tokenizer, FINETUNE_MAX_SEQ_LEN)
    val_dataset = TripletDataset(val_df, tokenizer, FINETUNE_MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    
    # 5. Optimizer e Loss
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=FINETUNE_LR)
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Il modello è già su device_map="auto", ma assicuriamoci
    
    # 6. Training Loop
    logger.info("Inizio Training Loop...")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Prepariamo gli input (squeeze per rimuovere dim in più di DataLoader)
            anchor_inputs = {k: v.squeeze(1).to(device) for k, v in batch["anchor"].items()}
            positive_inputs = {k: v.squeeze(1).to(device) for k, v in batch["positive"].items()}
            negative_inputs = {k: v.squeeze(1).to(device) for k, v in batch["negative"].items()}
            
            # Forward pass
            anchor_out = mean_pooling(model(**anchor_inputs), anchor_inputs["attention_mask"])
            positive_out = mean_pooling(model(**positive_inputs), positive_inputs["attention_mask"])
            negative_out = mean_pooling(model(**negative_inputs), negative_inputs["attention_mask"])
            
            # Normalizziamo embeddings
            anchor_out = F.normalize(anchor_out, p=2, dim=1)
            positive_out = F.normalize(positive_out, p=2, dim=1)
            negative_out = F.normalize(negative_out, p=2, dim=1)
            
            loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            
            # Backward
            loss.backward()
            
            if (step + 1) % FINETUNE_GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_train_loss += loss.item()
            
            if step % 1000 == 0:
                logger.info(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} - Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                anchor_inputs = {k: v.squeeze(1).to(device) for k, v in batch["anchor"].items()}
                positive_inputs = {k: v.squeeze(1).to(device) for k, v in batch["positive"].items()}
                negative_inputs = {k: v.squeeze(1).to(device) for k, v in batch["negative"].items()}
                
                anchor_out = F.normalize(mean_pooling(model(**anchor_inputs), anchor_inputs["attention_mask"]), p=2, dim=1)
                positive_out = F.normalize(mean_pooling(model(**positive_inputs), positive_inputs["attention_mask"]), p=2, dim=1)
                negative_out = F.normalize(mean_pooling(model(**negative_inputs), negative_inputs["attention_mask"]), p=2, dim=1)
                
                loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"--- Epoch {epoch+1} completata. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 7. Salvataggio Adapter
    logger.info(f"Salvataggio LoRA adapter in {ROBERTA_ADAPTER_DIR}")
    model.save_pretrained(ROBERTA_ADAPTER_DIR)
    tokenizer.save_pretrained(ROBERTA_ADAPTER_DIR)
    logger.info("Fine-Tuning completato.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_roberta_lora()
