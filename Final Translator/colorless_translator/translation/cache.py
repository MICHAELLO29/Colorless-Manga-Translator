"""Translation caching system for API call reduction."""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict


class TranslationCache:
    """Persistent cache for translations to reduce API calls."""
    
    def __init__(self, cache_file: str = "translation_cache.json", max_size: int = 10000):
        self.cache_file = Path(cache_file)
        self.max_size = max_size
        self.cache: Dict = {}
        self.hits = 0
        self.misses = 0
        self._load()
    
    def _hash_key(self, text: str, strategy: str = "standard") -> str:
        """Generate cache key from text and strategy."""
        key = f"{text}|{strategy}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    def _load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached translations from {self.cache_file}")
            except Exception as e:
                print(f"Could not load cache: {e}. Starting with empty cache.")
                self.cache = {}
        else:
            print("No cache file found. Starting with empty cache.")
    
    def save(self):
        """Save cache to disk."""
        try:
            if len(self.cache) > self.max_size:
                items = list(self.cache.items())
                self.cache = dict(items[-self.max_size:])
            
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Could not save cache: {e}")
    
    def get(self, text: str, strategy: str = "standard") -> Optional[str]:
        """Get cached translation if available."""
        key = self._hash_key(text, strategy)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]["translation"]
        self.misses += 1
        return None
    
    def set(self, text: str, translation: str, strategy: str = "standard"):
        """Store translation in cache."""
        key = self._hash_key(text, strategy)
        self.cache[key] = {
            "text": text,
            "translation": translation,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        if stats["hits"] + stats["misses"] > 0:
            print(f"\nTranslation Cache Stats:")
            print(f"   Cache size: {stats['size']} entries")
            print(f"   Cache hits: {stats['hits']} (saved API calls!)")
            print(f"   Cache misses: {stats['misses']}")
            print(f"   Hit rate: {stats['hit_rate']:.1f}%")
            if stats["hits"] > 0:
                print(f"   Estimated savings: ~{stats['hits']} API requests")
