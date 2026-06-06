import polars as pl
import numpy as np
import os
import torch
import logging
from config import PARQUET_5CORE_PATH, EMBEDDINGS_DIR, SBERT_MODEL_NAME, EMBEDDING_BATCH_SIZE, ROBERTA_MODEL_NAME, ROBERTA_ADAPTER_DIR, EMBEDDINGS_ROBERTA_DIR, FINETUNE_MAX_SEQ_LEN
from src.preprocessing import clean_text_columns

logger = logging.getLogger(__name__)

def generate_embeddings(batch_texts: list[str], model, device) -> np.ndarray:
    """Codifica un batch di stringhe usando il modello."""
    with torch.no_grad():
        embeddings = model.encode(batch_texts, batch_size=len(batch_texts), device=device, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def build_profiles(df: pl.DataFrame, embeddings: np.ndarray) -> tuple[dict, dict]:
    """Costruisce i profili medi per utente e item."""
    logger.info("Costruzione profili (medie embeddings)...")
    
    users = df["user_id"].to_numpy()
    items = df["parent_asin"].to_numpy()
    
    import collections
    
    user_idxs = collections.defaultdict(list)
    item_idxs = collections.defaultdict(list)
    
    for i, (u, it) in enumerate(zip(users, items)):
        user_idxs[u].append(i)
        item_idxs[it].append(i)
        
    user_profiles = {}
    for u, idxs in user_idxs.items():
        user_profiles[u] = np.mean(embeddings[idxs], axis=0)
        
    item_profiles = {}
    for it, idxs in item_idxs.items():
        item_profiles[it] = np.mean(embeddings[idxs], axis=0)
        
    logger.info(f"Creati {len(user_profiles)} user profiles e {len(item_profiles)} item profiles.")
    return user_profiles, item_profiles

def run_embedding_pipeline(model_type="sbert"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "sbert":
        from sentence_transformers import SentenceTransformer
        logger.info(f"Caricando modello: {SBERT_MODEL_NAME} su {device}")
        model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        out_dir = EMBEDDINGS_DIR
    elif model_type == "roberta":
        # Check if we need to load PEFT
        if not os.path.exists(ROBERTA_ADAPTER_DIR):
            logger.error(f"LoRA adapter non trovato in {ROBERTA_ADAPTER_DIR}. Eseguire fine-tuning prima.")
            return
            
        logger.info(f"Caricando RoBERTa + LoRA da {ROBERTA_ADAPTER_DIR}...")
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel
        
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_ADAPTER_DIR)
        base_model = AutoModel.from_pretrained(ROBERTA_MODEL_NAME, torch_dtype=torch.float16, device_map="auto", add_pooling_layer=False)
        model = PeftModel.from_pretrained(base_model, ROBERTA_ADAPTER_DIR)
        model.eval()
        out_dir = EMBEDDINGS_ROBERTA_DIR
    else:
        raise ValueError("model_type deve essere 'sbert' o 'roberta'")
    
    logger.info("Caricamento dataset e pulizia testuale...")
    df = pl.read_parquet(PARQUET_5CORE_PATH)
    df = clean_text_columns(df)
    
    texts = df["review_text"].to_list()
    
    logger.info(f"Produzione embeddings per {len(texts)} recensioni in batch...")
    
    if model_type == "sbert":
        all_embeddings = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, device=device, show_progress_bar=True, convert_to_numpy=True)
    else:
        # Batching manuale per RoBERTa
        import torch.nn.functional as F
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        emb_file = os.path.join(out_dir, "reviews_embeddings.npy")
        checkpoint_file = os.path.join(out_dir, "checkpoint_embed_roberta.txt")
        
        # Determina la dimensione dell'embedding con un piccolo batch di prova
        logger.info("Rilevamento dimensione dell'embedding di RoBERTa...")
        sample_inputs = tokenizer(["test text"], padding="max_length", truncation=True, max_length=FINETUNE_MAX_SEQ_LEN, return_tensors="pt")
        sample_inputs = {k: v.to(model.device) for k, v in sample_inputs.items()}
        with torch.no_grad():
            sample_outputs = model(**sample_inputs)
            sample_emb = mean_pooling(sample_outputs, sample_inputs["attention_mask"])
            embedding_dim = sample_emb.shape[1]
        logger.info(f"Dimensione dell'embedding rilevata: {embedding_dim}")

        # Controlliamo se esiste già un checkpoint e il file parziale
        start_idx = 0
        if os.path.exists(checkpoint_file) and os.path.exists(emb_file):
            try:
                with open(checkpoint_file, "r") as f:
                    start_idx = int(f.read().strip())
                logger.info(f"Checkpoint trovato. Ripristino dell'elaborazione dall'indice {start_idx}...")
                mode = 'r+'
            except Exception as e:
                logger.warning(f"Errore nel caricamento del checkpoint, riavvio da 0: {e}")
                start_idx = 0
                mode = 'w+'
        else:
            mode = 'w+'
            
        logger.info(f"Inizializzazione file di output in formato memmap .npy con modalità '{mode}'...")
        all_embeddings = np.lib.format.open_memmap(emb_file, mode=mode, dtype='float32', shape=(len(texts), embedding_dim))
        
        chunk_size = 100000
        batch_size = 512 # Aumentato per sfruttare meglio la RTX 5070 in float16
        
        for chunk_start in range(start_idx, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            logger.info(f"Tokenizzazione chunk {chunk_start}-{chunk_end}...")
            tokenized_chunk = tokenizer(chunk_texts, padding="max_length", truncation=True, max_length=FINETUNE_MAX_SEQ_LEN, return_tensors="pt")
            
            logger.info(f"Inference GPU per il chunk...")
            for i in range(0, len(chunk_texts), batch_size):
                batch_start = chunk_start + i
                batch_end = min(batch_start + batch_size, chunk_end)
                
                # Fetta del batch pre-tokenizzato
                inputs = {k: v[i:i+(batch_end-batch_start)].to(model.device) for k, v in tokenized_chunk.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = mean_pooling(outputs, inputs["attention_mask"])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                all_embeddings[batch_start:batch_end] = embeddings.cpu().numpy()
                
            # Salva il checkpoint e fai il flush a fine di ogni chunk
            all_embeddings.flush()
            with open(checkpoint_file, "w") as f:
                f.write(str(chunk_end))
            logger.info(f"  Elaborati {chunk_end}/{len(texts)} testi...")
                
        # Flush finale
        all_embeddings.flush()
        logger.info(f"Embeddings salvati in {emb_file}")
        
        # Rimuoviamo il checkpoint a fine esecuzione
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
            except Exception as e:
                logger.warning(f"Impossibile rimuovere il file di checkpoint: {e}")
    
    user_profiles, item_profiles = build_profiles(df, all_embeddings)
    
    np.save(os.path.join(out_dir, "user_profiles.npy"), user_profiles, allow_pickle=True)
    np.save(os.path.join(out_dir, "item_profiles.npy"), item_profiles, allow_pickle=True)
    logger.info("Pipeline completata.")

def run_roberta_embedding_pipeline():
    run_embedding_pipeline(model_type="roberta")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_embedding_pipeline()
