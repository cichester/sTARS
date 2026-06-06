import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import GEMINI_MODEL_NAME, GEMINI_CACHE_PATH, GENAI_RPM_LIMIT

logger = logging.getLogger(__name__)

# Carica le variabili dal file .env nella directory corrente
load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.cache = self._load_cache()
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Client Gemini caricato con modello: {GEMINI_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Errore inizializzazione client Gemini: {e}")
        else:
            logger.warning("Variabile d'ambiente GEMINI_API_KEY non trovata. Le chiamate all'API falliranno e useranno i fallback.")

    def _load_cache(self) -> dict:
        if os.path.exists(GEMINI_CACHE_PATH):
            try:
                with open(GEMINI_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Impossibile leggere la cache in {GEMINI_CACHE_PATH}: {e}. Creo una nuova cache.")
        return {}

    def _save_cache(self):
        try:
            # Assicuriamoci che la cartella esista
            os.makedirs(os.path.dirname(GEMINI_CACHE_PATH), exist_ok=True)
            with open(GEMINI_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Errore nel salvataggio della cache: {e}")

    def generate_reviews(self, product_id: str, count: int, sentiment: str) -> list[dict]:
        """
        Genera 'count' recensioni per un certo product_id con un dato sentiment.
        Controlla prima la cache su disco, poi l'API Gemini. In caso di errore, usa un fallback deterministico.
        """
        cache_key = f"{product_id}_{sentiment}"
        
        # Se abbiamo abbastanza recensioni in cache, le restituiamo direttamente
        if cache_key in self.cache and len(self.cache[cache_key]) >= count:
            logger.debug(f"Recuperate {count} recensioni ({sentiment}) per {product_id} dalla cache.")
            return self.cache[cache_key][:count]

        # Altrimenti dobbiamo generare. Generiamo almeno 5 per fare pre-caching di futuri filler/bot
        generate_count = max(count, 5)

        if not self.client:
            logger.debug(f"Nessun client Gemini disponibile per {product_id}. Uso il fallback.")
            fallback = self._generate_fallback(product_id, generate_count, sentiment)
            self.cache[cache_key] = fallback
            return fallback[:count]

        rating_range = "4.0 to 5.0" if sentiment == "positive" else "3.0 to 4.0" if sentiment == "neutral" else "1.0 to 2.0"
        
        prompt = f"""
You are simulating realistic Amazon customer reviews for an electronics product (ASIN: {product_id}).
Generate {generate_count} unique, believable product reviews.
Guidelines:
1. Sentiment: {sentiment} (rating range: {rating_range})
2. Each review must have a different writing style, length (15-100 words), and customer perspective.
3. Include natural human imperfections: occasional minor typos, informal language, varied punctuation.
4. Some reviews should be short, others slightly longer. Do not make them sound like professional marketing copy.
5. Provide a corresponding review title.

Output format: Return ONLY a valid JSON array of objects, where each object has exactly two fields:
"title" (string) and "text" (string).
Example:
[
  {{"title": "Does the job", "text": "Bought this last week. It works fine so far, but the shipping was slow."}},
  {{"title": "Awesome!", "text": "this thing is amazing! definitely worth the money. will buy again."}}
]
Do not include markdown tags, code blocks, or extra text. Output only raw JSON.
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Invio richiesta a Gemini ({GEMINI_MODEL_NAME}) per generare {generate_count} recensioni ({sentiment}) per {product_id} (tentativo {attempt+1}/{max_retries})...")
                
                # Chiamata al nuovo SDK di Google Gemini
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.85
                    )
                )
                
                # Rispettiamo il rate limiting (RPM)
                time.sleep(60.0 / GENAI_RPM_LIMIT)
                
                res_text = response.text.strip() if response.text else ""
                
                # Pulizia per evitare errori di markdown se il modello non rispetta application/json
                if res_text.startswith("```json"):
                    res_text = res_text[7:]
                if res_text.endswith("```"):
                    res_text = res_text[:-3]
                res_text = res_text.strip()
                
                reviews = json.loads(res_text)
                if isinstance(reviews, list) and len(reviews) > 0:
                    self.cache[cache_key] = reviews
                    self._save_cache()
                    return reviews[:count]
                else:
                    logger.warning(f"La risposta non contiene un array valido: {res_text[:200]}")
            except Exception as e:
                logger.error(f"Errore durante la generazione tramite Gemini: {e}")
                time.sleep(2)

        logger.warning(f"Impossibile generare le recensioni tramite Gemini. Uso il fallback per {product_id}.")
        fallback = self._generate_fallback(product_id, generate_count, sentiment)
        self.cache[cache_key] = fallback
        return fallback[:count]

    def _generate_fallback(self, product_id: str, count: int, sentiment: str) -> list[dict]:
        """Testi di fallback pronti se le API falliscono o non sono configurate."""
        fallback_texts = []
        for i in range(count):
            if sentiment == "positive":
                fallback_texts.append({
                    "title": f"Excellent product! {i}",
                    "text": f"This item {product_id} works perfectly. Had no issues setting it up and it does exactly what I needed. Highly recommended."
                })
            elif sentiment == "neutral":
                fallback_texts.append({
                    "title": f"It is okay {i}",
                    "text": f"The item {product_id} is decent for the price. Works okay, nothing special but it is not bad either."
                })
            else:
                fallback_texts.append({
                    "title": f"Very bad quality {i}",
                    "text": f"Very disappointed with this {product_id}. Stopped working after two days, would not recommend to anyone."
                })
        return fallback_texts
