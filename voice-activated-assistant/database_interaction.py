import asyncio
import aiosqlite
import logging

class ChatDatabase:
    def __init__(self, database_path):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    async def init_db(self):
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.executescript('''
                    CREATE TABLE IF NOT EXISTS Responses (
                        pattern TEXT NOT NULL,
                        response TEXT NOT NULL,
                        frequency INTEGER DEFAULT 1,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (pattern, response)
                    );
                    CREATE TABLE IF NOT EXISTS Contexts (
                        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        context_key TEXT NOT NULL,
                        context_value TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                ''')
                await db.commit()
                self.logger.info("Database initialized and tables created.")
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    async def update_response_frequency(self, response):
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute('''
                    UPDATE Responses
                    SET frequency = frequency + 1, last_used = CURRENT_TIMESTAMP
                    WHERE response = ?
                ''', (response,))
                await db.commit()
                self.logger.info(f"Updated frequency for response: {response}")
        except Exception as e:
            self.logger.error(f"Error updating response frequency: {e}")
            raise

    async def close(self):
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.close()
                self.logger.info("Database connection closed successfully.")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
            raise

if __name__ == "__main__":
    database = ChatDatabase('chat_memory.db')
    asyncio.run(database.init_db())