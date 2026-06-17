from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = PROJECT_ROOT / ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )
    
    # 1. Projects Roots
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # 2. Environment Variables and Secrets
    ENVIRONMET: str = Field(default="development")
    
    # 3. ML Global Parameters
    RANDOM_SEED: int = 42
    
    # 4. Data Structure(For Ingestion)
    COLUMNS_TO_DROP: list[str] = ["Country", 'State', 'City', 'Zip Code', 'Total Revenue', 'Satisfaction Score', 'Quarter', 'Churn Score', 'Customer Status', 'Churn Category', 'Churn Reason']
    TARGET_COLUMN: str = 'Churn Label'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

config = Settings()