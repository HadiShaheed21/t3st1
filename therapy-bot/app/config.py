import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Settings class to manage environment variables securely.
    Access variables as attributes, e.g., settings.OPENAI_API_KEY
    """
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX")

    # Add more variables as needed

    def validate(self):
        # Ensure all required variables are set
        missing = [
            var for var in [
                ("OPENAI_API_KEY", self.OPENAI_API_KEY),
                ("PINECONE_API_KEY", self.PINECONE_API_KEY),
                ("PINECONE_ENVIRONMENT", self.PINECONE_ENVIRONMENT),
                ("PINECONE_INDEX", self.PINECONE_INDEX),
            ] if not var[1]
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join([m[0] for m in missing])}")

# Instantiate and validate settings at import time
settings = Settings()
settings.validate() 