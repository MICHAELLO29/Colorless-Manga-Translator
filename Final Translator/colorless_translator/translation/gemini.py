"""Gemini API translation service using the new google-genai SDK."""

import re
import time
from typing import List, Optional

from colorless_translator.core.exceptions import QuotaExhaustedError, TranslationError
from colorless_translator.translation.cache import TranslationCache


STRATEGY_HINTS = {
    "action": "Focus on impact and brevity. Sound effects can be transliterated (e.g., 'BOOM', 'SLASH'). Keep exclamations punchy.",
    "dialogue": "Focus on natural conversation flow. Use contractions heavily. Match speaking patterns (casual, formal, excited, etc.).",
    "standard": "Balance natural dialogue with action text. Adapt tone based on context.",
}


class GeminiTranslator:
    """Handles translation via Google Gemini API with caching and retry logic.
    
    Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).
    Supports auto model selection: set model_name='auto' to probe and pick the best.
    """

    # Ranked by translation quality (best first)
    MODEL_TIERS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "auto",
        cache: Optional[TranslationCache] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        max_length_ratio: float = 1.8,
    ):
        self.cache = cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_length_ratio = max_length_ratio
        self.api_key = api_key
        
        self._configure_api(api_key)

        if model_name == "auto":
            self.model_name = self._auto_pick_model()
        else:
            self.model_name = model_name

        self._model_recovered = False
    
    def _configure_api(self, api_key: str):
        """Configure Gemini API using the new google-genai SDK."""
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def _auto_pick_model(self) -> str:
        """Probe available models and pick the best one for translation."""
        print("--- Auto Model Picker: Probing available models ---")

        # 1. List models that support generateContent
        try:
            models_response = self.client.models.list()
            available = []
            for m in list(models_response):
                name = getattr(m, "name", "") or ""
                methods = getattr(m, "supported_generation_methods", None) or []
                methods = [str(x) for x in methods]
                if "generateContent" in methods and name:
                    short = name.replace("models/", "")
                    available.append(short)
        except Exception as e:
            print(f"   Could not list models ({e}), defaulting to gemini-2.5-flash")
            return "gemini-2.5-flash"

        if not available:
            print("   No eligible models found, defaulting to gemini-2.5-flash")
            return "gemini-2.5-flash"

        print(f"   Found {len(available)} eligible models")

        # 2. Rank by tier preference
        ranked = []
        for tier_model in self.MODEL_TIERS:
            for avail in available:
                if avail.startswith(tier_model):
                    ranked.append(avail)
        # Add any remaining models not in our tier list
        for avail in available:
            if avail not in ranked:
                ranked.append(avail)

        # 3. Test top candidates with a quick translation probe
        for candidate in ranked[:5]:  # Test up to 5 candidates
            try:
                test_response = self.client.models.generate_content(
                    model=candidate,
                    contents='Translate this Japanese to English in 3 words or less: "こんにちは"',
                )
                if test_response and test_response.text:
                    print(f"   Selected model: {candidate}")
                    return candidate
            except Exception as e:
                print(f"   Model {candidate} failed: {str(e)[:60]}")
                continue

        # 4. Fallback
        fallback = ranked[0] if ranked else "gemini-2.5-flash"
        print(f"   Falling back to: {fallback}")
        return fallback

    def _generate_content(self, prompt: str) -> str:
        """Generate content using the Gemini API client."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    def _is_model_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return (
            "not supported" in msg
            or "unsupported" in msg
            or "is not found" in msg
            or "404" in msg
            or "not found" in msg
            or "deprecated" in msg
            or "decommissioned" in msg
        )

    def _recover_model(self) -> bool:
        """Try to find a working model for generateContent and swap to it."""
        try:
            models_response = self.client.models.list()
            models = list(models_response)
        except Exception:
            return False

        eligible = []
        for m in models:
            name = getattr(m, "name", "") or ""
            methods = getattr(m, "supported_generation_methods", None) or []
            methods = [str(x) for x in methods]
            if "generateContent" not in methods:
                continue
            if not name:
                continue
            eligible.append(name)

        if not eligible:
            return False

        # Prefer current stable models (as of 2026)
        preferred = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-2.5-pro",
        ]

        chosen = None
        for p in preferred:
            if p in eligible:
                chosen = p
                break
        if chosen is None:
            # Pick the first eligible model
            chosen = eligible[0]

        self.model_name = chosen.replace("models/", "")
        print(f"   Recovered to model: {self.model_name}")
        return True
    
    def translate_batch(
        self,
        texts: List[str],
        strategy: str = "standard",
        retry_count: int = 0,
    ) -> List[str]:
        """
        Translate multiple texts in a single API call.
        
        Args:
            texts: List of Japanese texts to translate
            strategy: Translation strategy (action/dialogue/standard)
            retry_count: Current retry attempt
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        cached, uncached_texts, uncached_indices = self._check_cache(texts, strategy)
        
        if not uncached_texts:
            print(f"   All {len(texts)} translations from cache (0 API calls!)")
            return [t for _, t in sorted(cached)]
        
        if cached:
            print(f"   {len(cached)}/{len(texts)} translations from cache")
        
        try:
            new_translations = self._translate_via_api(uncached_texts, strategy)
            
            if len(new_translations) == len(uncached_texts):
                self._store_in_cache(uncached_texts, new_translations, strategy)
                return self._merge_translations(texts, cached, uncached_indices, new_translations)
            else:
                print(f"    Translation count mismatch. Retrying individually...")
                return [self.translate_single(t, strategy) for t in texts]
                
        except Exception as e:
            return self._handle_batch_error(e, texts, strategy, retry_count)
    
    def translate_single(
        self,
        text: str,
        strategy: str = "standard",
        retry_count: int = 0,
    ) -> str:
        """Translate a single text with caching."""
        if self.cache:
            cached = self.cache.get(text, strategy)
            if cached:
                cleaned_cached = self._clean_translation(cached)
                if cleaned_cached != cached:
                    self.cache.set(text, cleaned_cached, strategy)
                return cleaned_cached

        hint = STRATEGY_HINTS.get(strategy, STRATEGY_HINTS["standard"])
        prompt = f"""Expert manga translator. Translate Japanese to natural English.

## Context: {strategy.upper()} manga page
{hint}

## Rules:
1. CONCISE - Fit speech bubbles (max 1.5x original length)
2. NATURAL - Use contractions where natural
3. PRESERVE - Tone, emotion, personality
4. FORMAT - Output ONLY the final translation text
5. NO OPTIONS - No alternatives, no lists, no explanations, no notes

Japanese: "{text}"
English:"""
        
        try:
            raw_response = self._generate_content(prompt)
            translation = self._clean_translation(raw_response)

            if self.cache:
                self.cache.set(text, translation, strategy)

            return translation
            
        except Exception as e:
            return self._handle_single_error(e, text, strategy, retry_count)
    
    def _check_cache(self, texts: List[str], strategy: str):
        """Check cache for existing translations."""
        cached = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.cache:
                result = self.cache.get(text, strategy)
                if result:
                    cleaned = self._clean_translation(result)
                    if cleaned != result:
                        self.cache.set(text, cleaned, strategy)
                    cached.append((i, cleaned))
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        return cached, uncached_texts, uncached_indices
    
    def _translate_via_api(self, texts: List[str], strategy: str) -> List[str]:
        """Execute batch translation via API."""
        numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
        hint = STRATEGY_HINTS.get(strategy, STRATEGY_HINTS["standard"])
        
        prompt = f"""Expert manga translator. Translate Japanese to natural English.

## Context: {strategy.upper()} manga page
{hint}

## Rules:
1. CONCISE - Fit speech bubbles (max 1.5x original length)
2. NATURAL - Use contractions, informal speech
3. PRESERVE - Tone, emotion, personality
4. ADAPT - Idioms, cultural references naturally
5. FORMAT - Separate with '|||' ONLY
6. COUNT - Output {len(texts)} translations (no more, no less)
7. FINAL ONLY - Output ONLY the final translation, NO alternatives, NO explanations, NO notes

## Common Manga Terms:
- さん/kun/chan → Use names directly or Mr./Ms. only if formal
- はい/うん → Yeah/Yep/Yes (match formality)
- すごい → Wow/Amazing/Incredible (match intensity)
- やった → Yes!/Alright!/Got it! (match emotion)
- ちょっと → Hey/Wait/Hold on (match context)

## Japanese Text:
{numbered}

## English (FINAL translation only, separate with '|||'):"""
        
        raw_response = self._generate_content(prompt)
        raw = [t.strip() for t in raw_response.split("|||")]
        
        return [self._clean_translation(t) for t in raw]
    
    def _clean_translation(self, text: str) -> str:
        """Remove alternatives and explanations from translation."""
        cleaned = text or ""
        cleaned = re.sub(r'\([^)]*[Aa][Ll][Tt][Ee][Rr][Nn][Aa][Tt][Ii][Vv][Ee][^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*[Oo][Pp][Tt][Ii][Oo][Nn][Ss][^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*[Nn][Oo][Tt][Ee][^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*\s+[Oo][Rr]\s+[^)]*\)', '', cleaned)
        cleaned = cleaned.replace("*", "")
        cleaned = cleaned.replace("_", "")

        if "|||" in cleaned:
            cleaned = cleaned.split("|||", 1)[0]

        if re.search(r"\b(here are|here's)\b.*\b(options|choices)\b", cleaned, flags=re.IGNORECASE) or re.search(
            r"\bdepending on\b.*\bnuance\b", cleaned, flags=re.IGNORECASE
        ):
            quoted = re.findall(r"[\"\u201c\u201d](.+?)[\"\u201c\u201d]", cleaned)
            quoted = [q.strip() for q in quoted if q.strip()]
            if quoted:
                cleaned = quoted[0]
            else:
                parts = cleaned.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    cleaned = parts[1].strip()
                cleaned = re.split(r"\s+(?:OR|/|\\|;)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if lines:
            drop_prefix = (
                "here are",
                "here's",
                "options",
                "depending",
                "the exact nuance",
                "note:",
                "explanation",
            )
            filtered_lines = [
                ln for ln in lines
                if not ln.lower().startswith(drop_prefix)
                and not ln.lower().startswith("japanese:")
                and not ln.lower().startswith("english:")
            ]

            candidates: List[str] = []
            for ln in filtered_lines:
                m = re.match(r"^(?:\d+[\).\]]\s*|[-*]\s+)(.+)$", ln)
                if m:
                    candidates.append(m.group(1).strip())

            if candidates:
                cleaned = candidates[0]
            else:
                cleaned = filtered_lines[0] if filtered_lines else lines[0]

        cleaned = cleaned.strip().strip('"').strip("'")
        return " ".join(cleaned.split()).strip()
    
    def _store_in_cache(self, texts: List[str], translations: List[str], strategy: str):
        """Store translations in cache."""
        if self.cache:
            for text, trans in zip(texts, translations):
                self.cache.set(text, trans, strategy)
    
    def _merge_translations(
        self,
        original_texts: List[str],
        cached: List[tuple],
        uncached_indices: List[int],
        new_translations: List[str],
    ) -> List[str]:
        """Merge cached and new translations in correct order."""
        result = [None] * len(original_texts)
        
        for i, trans in cached:
            result[i] = trans
        
        for idx, trans in zip(uncached_indices, new_translations):
            result[idx] = trans
        
        for i, (orig, trans) in enumerate(zip(original_texts, result)):
            if trans and len(trans) / max(len(orig), 1) > self.max_length_ratio:
                print(f'    Block #{i+1} translation is long: "{trans[:50]}..."')
        
        return result
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is quota-related."""
        error_str = str(error).lower()
        keywords = ["quota", "rate limit", "resource exhausted", "429", "too many requests"]
        return any(kw in error_str for kw in keywords)

    def _is_overloaded_error(self, error: Exception) -> bool:
        """Check if error is a 503 model overload."""
        error_str = str(error).lower()
        keywords = ["503", "unavailable", "high demand", "overloaded", "service unavailable"]
        return any(kw in error_str for kw in keywords)

    def _fallback_to_next_model(self) -> bool:
        """Switch to the next-tier model when the current one is overloaded."""
        current = self.model_name
        print(f"    Model {current} is overloaded. Trying next model...")

        # Find current tier index and try the next ones
        current_idx = -1
        for i, tier in enumerate(self.MODEL_TIERS):
            if current.startswith(tier):
                current_idx = i
                break

        candidates = self.MODEL_TIERS[current_idx + 1:] if current_idx >= 0 else self.MODEL_TIERS[1:]

        for candidate in candidates:
            try:
                test = self.client.models.generate_content(
                    model=candidate,
                    contents='Say "ok"',
                )
                if test and test.text:
                    print(f"    Switched to model: {candidate}")
                    self.model_name = candidate
                    return True
            except Exception:
                continue

        print("    No fallback model available.")
        return False
    
    def _handle_batch_error(
        self,
        error: Exception,
        texts: List[str],
        strategy: str,
        retry_count: int,
    ) -> List[str]:
        """Handle batch translation error with retry logic."""
        if self._is_model_error(error) and not self._model_recovered:
            if self._recover_model():
                self._model_recovered = True
                return self.translate_batch(texts, strategy, retry_count=retry_count)
        if self._is_overloaded_error(error):
            if retry_count < 1:
                wait = self.retry_delay
                print(f"    Model overloaded (503). Waiting {wait}s before retry...")
                time.sleep(wait)
                return self.translate_batch(texts, strategy, retry_count + 1)
            else:
                # Model stays overloaded — switch to next tier
                if self._fallback_to_next_model():
                    return self.translate_batch(texts, strategy, retry_count=0)
                print(f"    All models overloaded. Returning errors.")
                return ["[Translation Error]" for _ in texts]
        if self._is_quota_error(error):
            if retry_count < self.max_retries:
                wait = self.retry_delay * (2 ** retry_count)
                print(f"    API rate limit hit. Waiting {wait}s before retry {retry_count + 1}/{self.max_retries}...")
                time.sleep(wait)
                return self.translate_batch(texts, strategy, retry_count + 1)
            else:
                print(f"\nAPI QUOTA EXHAUSTED: {error}")
                raise QuotaExhaustedError("API quota exhausted") from error
        else:
            print(f"    Translation error: {error}. Falling back to individual translation.")
            return [self.translate_single(t, strategy) for t in texts]
    
    def _handle_single_error(
        self,
        error: Exception,
        text: str,
        strategy: str,
        retry_count: int,
    ) -> str:
        """Handle single translation error with retry logic."""
        if self._is_model_error(error) and not self._model_recovered:
            if self._recover_model():
                self._model_recovered = True
                return self.translate_single(text, strategy, retry_count=retry_count)
        if self._is_overloaded_error(error):
            if retry_count < 1:
                wait = self.retry_delay
                print(f"    Model overloaded (503). Waiting {wait}s...")
                time.sleep(wait)
                return self.translate_single(text, strategy, retry_count + 1)
            else:
                if self._fallback_to_next_model():
                    return self.translate_single(text, strategy, retry_count=0)
                print(f'    All models overloaded for "{text[:30]}..."')
                return "[Translation Error]"
        if self._is_quota_error(error):
            if retry_count < self.max_retries:
                wait = self.retry_delay * (2 ** retry_count)
                print(f"    API rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                return self.translate_single(text, strategy, retry_count + 1)
            else:
                raise QuotaExhaustedError("API quota exhausted") from error
        else:
            print(f'    Error translating "{text[:30]}...": {error}')
            return "[Translation Error]"
