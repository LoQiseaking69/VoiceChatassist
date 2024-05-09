from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
import logging

class MLModel:
    def __init__(self, logger):
        self.logger = logger
        # Initialize the sentiment analysis model
        model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer_sentiment = AutoTokenizer.from_pretrained(model_name_sentiment)
        model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_name_sentiment)
        self.sentiment_analyzer = pipeline('sentiment-analysis', model=model_sentiment, tokenizer=tokenizer_sentiment)

        # Initialize the summarization model 
        model_name_summarization = "facebook/bart-large-cnn"
        tokenizer_summarization = AutoTokenizer.from_pretrained(model_name_summarization)
        model_summarization = AutoModelForSeq2SeqLM.from_pretrained(model_name_summarization)
        self.summarizer = pipeline('summarization', model=model_summarization, tokenizer=tokenizer_summarization)

    @lru_cache(maxsize=100)
    def analyze_sentiment(self, text):
        try:
            result = self.sentiment_analyzer(text)
            score = result[0]['score']
            label = result[0]['label']
            if label == 'NEGATIVE':
                score = -score
            return score
        except Exception as e:
            self.logger.log_error(f"Error in sentiment analysis: {e}")
            return None  # Return None for indeterminate sentiment in case of error

    @lru_cache(maxsize=50)
    def summarize_text(self, text):
        try:
            result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            self.logger.log_error(f"Error in text summarization: {e}")
            return text  # Return original text if summarization fails

if __name__ == "__main__":
    from custom_logger import CustomLogger  # Assuming CustomLogger is in custom_logger.py
    logger = CustomLogger("ml_system_logs.log")
    ml_model = MLModel(logger)
