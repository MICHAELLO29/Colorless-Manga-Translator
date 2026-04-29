import os
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def _load_env():
    """Load .env file for API key."""
    if HAS_DOTENV:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
    return False


@dataclass
class PathSettings:
    """File and directory paths."""
    font_path: str = "fonts/animeace2_reg.ttf"
    input_folder: str = "manga pages"
    output_folder: str = "output"
    yolo_model_path: str = "models/dabest.pt"


@dataclass
class DetectionSettings:
    """YOLO detection confidence thresholds."""
    # Local YOLO model thresholds
    conf_bubble: float = 0.20
    conf_text_bubble: float = 0.20
    conf_clean_text: float = 0.20
    conf_messy_text: float = 0.20
    use_multi_class_fusion: bool = True
    # Roboflow API minimum confidence (overrides the above when using cloud)
    roboflow_min_conf: float = 0.20

    def as_threshold_dict(self) -> dict:
        """Return thresholds as dictionary for detection functions."""
        return {
            "bubble": self.conf_bubble,
            "text_bubble": self.conf_text_bubble,
            "clean_text": self.conf_clean_text,
            "messy_text": self.conf_messy_text,
        }


@dataclass
class TranslationSettings:
    """Translation behavior settings."""
    font_size: int = 14
    max_translation_length_ratio: float = 1.8
    max_retries: int = 3
    retry_delay: int = 5
    gemini_model_name: str = "auto"


@dataclass
class CacheSettings:
    """Translation cache settings."""
    enabled: bool = True
    cache_file: str = "translation_cache.json"
    max_size: int = 10000


@dataclass
class RoboflowSettings:
    """Roboflow cloud detection settings."""
    use_roboflow: bool = False
    api_key: str = ""
    model_id: str = "bubble-text-detector-k5qgg/1"


@dataclass
class FeatureFlags:
    """Feature toggles for optional functionality."""
    enable_translation_alternatives: bool = False
    enable_series_memory: bool = True
    enable_advanced_typography: bool = True
    series_name: str = "current_manga"


@dataclass
class Settings:
    """Main configuration container aggregating all settings."""
    gemini_api_key: str = ""
    paths: PathSettings = field(default_factory=PathSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    roboflow: RoboflowSettings = field(default_factory=RoboflowSettings)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load API keys from .env, use hardcoded defaults for everything else."""
        _load_env()
        rf_key = os.getenv("ROBOFLOW_API_KEY", "")
        use_rf = bool(rf_key) and os.getenv("USE_ROBOFLOW", "true").lower() in ("1", "true", "yes")
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            roboflow=RoboflowSettings(
                use_roboflow=use_rf,
                api_key=rf_key,
                model_id=os.getenv("ROBOFLOW_MODEL_ID", "bubble-text-detector-k5qgg/1"),
            ),
        )
    
    def validate(self) -> list[str]:
        """Validate settings and return list of errors."""
        errors = []
        
        if not self.gemini_api_key or self.gemini_api_key == "your_api_key_here":
            errors.append("GEMINI_API_KEY is not set or invalid")
        
        if not Path(self.paths.font_path).exists():
            errors.append(f"Font file not found: {self.paths.font_path}")
        
        # Only require a local YOLO model when NOT using Roboflow
        if not self.roboflow.use_roboflow and not Path(self.paths.yolo_model_path).exists():
            errors.append(f"YOLO model not found: {self.paths.yolo_model_path}")
        
        return errors
    
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.paths.output_folder).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
