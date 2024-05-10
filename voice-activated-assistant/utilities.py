import logging
from logging.handlers import RotatingFileHandler
import os
from cryptography.fernet import Fernet

class EncryptedFileHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False, encryption_key=None):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        if encryption_key is None:
            raise ValueError("Encryption key must be provided for encrypted logging.")
        self.encryptor = Fernet(encryption_key)

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
        try:
            msg = self.format(record)
            encrypted_msg = self.encryptor.encrypt(msg.encode()) + b'\n'
            os.write(self.stream.fileno(), encrypted_msg)
        except Exception as e:
            self.handleError(record)

class CustomLogger:
    def __init__(self, log_file, max_bytes=1000000, backup_count=5, encryption_key=None):
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encryption_key = encryption_key
        self.setup_logging()

    def setup_logging(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.chmod(os.path.dirname(self.log_file), 0o700)
        if self.encryption_key:
            handler = EncryptedFileHandler(self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encryption_key=self.encryption_key)
        else:
            handler = RotatingFileHandler(self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def log_interaction(self, text, response):
        self.logger.info(f'Processed interaction: Text: "{text}" -> Response: "{response}"')

    def log_error(self, message):
        self.logger.error(message)

    def log_info(self, message):
        self.logger.info(message)

if __name__ == "__main__":
    # In practice, retrieve an existing encryption key securely
    key = Fernet.generate_key()
    logger = CustomLogger("secure_system_logs.log", encryption_key=key)
    logger.log_interaction("Hello", "Hi there!")
    logger.log_error("This is an error message.")
    logger.log_info("This is general information.")