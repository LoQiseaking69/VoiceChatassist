import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import EncoderDecoderModel, BertTokenizer
from functools import lru_cache
import logging

class MLModel:
    def __init__(self, logger):
        self.logger = logger
        self.nlp = self.load_spacy_model("en_core_web_trf")
        self.init_transformers_models()

    def load_spacy_model(self, model_name):
        try:
            nlp = spacy.load(model_name)
            self.logger.log_info(f"Loaded spaCy model {model_name}")
            return nlp
        except Exception as e:
            self.logger.log_error(f"Failed to load spaCy model {model_name}: {e}")
            raise

    def init_transformers_models(self):
        model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer_sentiment = AutoTokenizer.from_pretrained(model_name_sentiment)
        model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_name_sentiment)
        self.sentiment_analyzer = pipeline('sentiment-analysis', model=model_sentiment, tokenizer=tokenizer_sentiment)

        model_name_summarization = "facebook/bart-large-cnn"
        tokenizer_summarization = AutoTokenizer.from_pretrained(model_name_summarization)
        model_summarization = AutoModelForSeq2SeqLM.from_pretrained(model_name_summarization)
        self.summarizer = pipeline('summarization', model=model_summarization, tokenizer=tokenizer_summarization)

    def process_text(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentences = [sent.text for sent in doc.sents]

        entity_based_processing = {ent[1]: [] for ent in entities}
        for sentence in sentences:
            sentiment_result = self.analyze_sentiment(sentence)
            summary = self.summarize_text(sentence)
            for ent in entities:
                if ent[0] in sentence:
                    entity_based_processing[ent[1]].append((sentence, sentiment_result, summary))

        return self.process_entities(entity_based_processing)

    def process_entities(self, data):
        results = []
        for entity_type, instances in data.items():
            if entity_type in ['PERSON', 'ORG']:
                for instance in instances:
                    if instance[1] is not None and instance[1] < 0:  # Negative sentiment
                        refined_summary = self.refine_summary(instance[0], focus='negative')
                        results.append({'entity': entity_type, 'sentence': instance[0], 'sentiment': instance[1], 'summary': refined_summary})
                    else:
                        results.append({'entity': entity_type, 'sentence': instance[0], 'sentiment': instance[1], 'summary': instance[2]})
            else:
                results.extend({'entity': entity_type, 'sentence': inst[0], 'sentiment': inst[1], 'summary': inst[2]} for inst in instances)
        return results

    def refine_summary(self, text, focus):
        if focus == 'negative':
            # Adjust summarization parameters for a more focused response on negative aspects
            try:
                result = self.summarizer(text, max_length=100, min_length=20, do_sample=False)
                return result[0]['summary_text']
            except Exception as e:
                self.logger.log_error(f"Refined summarization failed: {e}")
                return text
        return text

    @lru_cache(maxsize=100)
    def analyze_sentiment(self, text):
        try:
            result = self.sentiment_analyzer(text)
            score = result[0]['score']
            label = result[0]['label']
            return -score if label == 'NEGATIVE' else score
        except Exception as e:
            self.logger.log_error(f"Error in sentiment analysis: {e}")
            return None

    @lru_cache(maxsize=50)
    def summarize_text(self, text):
        try:
            result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            self.logger.log_error(f"Error in text summarization: {e}")
            return text

if __name__ == "__main__":
    from custom_logger import CustomLogger
    logger = CustomLogger("ml_system_logs.log")
    ml_model = MLModel(logger)
    sample_text = "The quick brown fox jumps over the lazy dog. It was a good day."
    print(ml_model.process_text(sample_text))
