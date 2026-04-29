"""
Series Memory System
Maintains context across pages for consistent translations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class SeriesMemory:
    """Remember context across pages in a manga series."""
    
    def __init__(self, series_name: str = "default", memory_file: str = "series_memory.json"):
        self.series_name = series_name
        self.memory_file = Path(memory_file)
        
        # Core memory components
        self.character_names = {}  # Japanese -> English mappings
        self.recurring_phrases = {}  # Japanese -> English for common phrases
        self.style_preferences = {}  # Translation style choices
        self.page_context = []  # Context from recent pages
        self.metadata = {
            'series_name': series_name,
            'total_pages': 0,
            'last_updated': None
        }
        
        # Load existing memory
        self.load_memory()
    
    def load_memory(self):
        """Load series memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load data for this series
                if self.series_name in data:
                    series_data = data[self.series_name]
                    self.character_names = series_data.get('character_names', {})
                    self.recurring_phrases = series_data.get('recurring_phrases', {})
                    self.style_preferences = series_data.get('style_preferences', {})
                    self.page_context = series_data.get('page_context', [])
                    self.metadata = series_data.get('metadata', self.metadata)
                    
                    print(f"Loaded series memory: {self.series_name}")
                    print(f"   Characters: {len(self.character_names)}, Phrases: {len(self.recurring_phrases)}")
            except Exception as e:
                print(f"Could not load series memory: {e}")
        else:
            print(f"Starting new series memory: {self.series_name}")
    
    def save_memory(self):
        """Save series memory to disk."""
        try:
            # Load existing data
            all_data = {}
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            
            # Update this series
            all_data[self.series_name] = {
                'character_names': self.character_names,
                'recurring_phrases': self.recurring_phrases,
                'style_preferences': self.style_preferences,
                'page_context': self.page_context[-20:],  # Keep last 20 pages
                'metadata': self.metadata
            }
            
            # Save
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved series memory: {self.series_name}")
        except Exception as e:
            print(f"Could not save series memory: {e}")
    
    def add_page_context(
        self, 
        page_num: int, 
        detected_texts: List[str],
        translations: List[str],
        strategy: str
    ):
        """Add context from a processed page."""
        
        # Extract character names from this page
        page_names = self._extract_names_from_texts(detected_texts)
        
        # Extract recurring phrases
        page_phrases = self._extract_phrases(detected_texts, translations)
        
        # Store page context
        page_data = {
            'page': page_num,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'character_names': page_names,
            'key_phrases': page_phrases,
            'text_count': len(detected_texts)
        }
        
        self.page_context.append(page_data)
        
        # Update metadata
        self.metadata['total_pages'] = max(self.metadata['total_pages'], page_num)
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        # Keep only last 20 pages in memory
        if len(self.page_context) > 20:
            self.page_context = self.page_context[-20:]
    
    def get_context_for_page(self, page_num: int, lookback: int = 5) -> Dict:
        """
        Get relevant context for translating a page.
        
        Args:
            page_num: Current page number
            lookback: How many previous pages to consider
        
        Returns:
            Dict with relevant context
        """
        # Get recent pages
        recent_pages = [p for p in self.page_context if p['page'] < page_num]
        recent_pages = recent_pages[-lookback:]
        
        # Aggregate context
        context = {
            'character_names': self.character_names.copy(),
            'recurring_phrases': self.recurring_phrases.copy(),
            'recent_strategies': [p['strategy'] for p in recent_pages],
            'recent_pages': recent_pages,
            'series_name': self.series_name
        }
        
        return context
    
    def register_character_name(self, japanese_name: str, english_name: str):
        """Register a character name for consistency."""
        self.character_names[japanese_name] = english_name
        print(f"   Registered character: {japanese_name} -> {english_name}")
    
    def register_recurring_phrase(self, japanese_phrase: str, english_phrase: str):
        """Register a recurring phrase for consistency."""
        self.recurring_phrases[japanese_phrase] = english_phrase
    
    def get_character_name(self, japanese_name: str) -> Optional[str]:
        """Get consistent English name for a character."""
        return self.character_names.get(japanese_name)
    
    def get_recurring_phrase(self, japanese_phrase: str) -> Optional[str]:
        """Get consistent translation for a recurring phrase."""
        return self.recurring_phrases.get(japanese_phrase)
    
    def _extract_names_from_texts(self, texts: List[str]) -> List[str]:
        """Extract potential character names from texts."""
        import re
        
        names = []
        honorifics = ['さん', 'kun', 'chan', '様', 'さま', '先生', 'せんせい']
        
        for text in texts:
            # Look for katakana names
            katakana_pattern = r'[\u30A0-\u30FF]{2,}'
            katakana_matches = re.findall(katakana_pattern, text)
            names.extend(katakana_matches)
            
            # Look for names with honorifics
            for honorific in honorifics:
                pattern = f'([\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+){honorific}'
                matches = re.findall(pattern, text)
                names.extend(matches)
        
        return list(set(names))
    
    def _extract_phrases(self, japanese_texts: List[str], english_texts: List[str]) -> Dict[str, str]:
        """Extract potential recurring phrases."""
        phrases = {}
        
        # Look for short, common phrases
        for jp, en in zip(japanese_texts, english_texts):
            if len(jp) <= 10 and len(en) <= 20:  # Short phrases
                phrases[jp] = en
        
        return phrases
    
    def get_statistics(self) -> Dict:
        """Get statistics about the series memory."""
        return {
            'series_name': self.series_name,
            'total_pages': self.metadata['total_pages'],
            'character_count': len(self.character_names),
            'phrase_count': len(self.recurring_phrases),
            'context_pages': len(self.page_context),
            'last_updated': self.metadata.get('last_updated', 'Never')
        }
    
    def print_statistics(self):
        """Print series memory statistics."""
        stats = self.get_statistics()
        print(f"\nSeries Memory Statistics:")
        print(f"   Series: {stats['series_name']}")
        print(f"   Total pages: {stats['total_pages']}")
        print(f"   Characters tracked: {stats['character_count']}")
        print(f"   Recurring phrases: {stats['phrase_count']}")
        print(f"   Context pages in memory: {stats['context_pages']}")
        
        if self.character_names:
            print(f"\n   Known characters:")
            for jp, en in list(self.character_names.items())[:5]:
                print(f"      {jp} -> {en}")
            if len(self.character_names) > 5:
                print(f"      ... and {len(self.character_names) - 5} more")
    
    def clear_memory(self):
        """Clear all memory for this series."""
        self.character_names = {}
        self.recurring_phrases = {}
        self.style_preferences = {}
        self.page_context = []
        self.metadata['total_pages'] = 0
        print(f"Cleared memory for series: {self.series_name}")
    
    def export_memory(self, export_path: str):
        """Export series memory to a separate file."""
        export_data = {
            'series_name': self.series_name,
            'character_names': self.character_names,
            'recurring_phrases': self.recurring_phrases,
            'metadata': self.metadata
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported series memory to: {export_path}")
    
    def import_memory(self, import_path: str):
        """Import series memory from a file."""
        with open(import_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.character_names = data.get('character_names', {})
        self.recurring_phrases = data.get('recurring_phrases', {})
        self.metadata = data.get('metadata', self.metadata)
        
        print(f"Imported series memory from: {import_path}")
        print(f"   Characters: {len(self.character_names)}, Phrases: {len(self.recurring_phrases)}")


class ContextAwareTranslator:
    """Wrapper for translation with series context.
    
    Uses the new google-genai Client pattern.
    """
    
    def __init__(self, gemini_client, model_name: str, series_memory: SeriesMemory):
        self.client = gemini_client
        self.model_name = model_name
        self.series_memory = series_memory
    
    def translate_with_context(
        self, 
        japanese: str, 
        strategy: str,
        page_num: int
    ) -> str:
        """Translate using series context for consistency."""
        
        # Get context
        context = self.series_memory.get_context_for_page(page_num)
        
        # Check for known character names
        for jp_name, en_name in context['character_names'].items():
            if jp_name in japanese:
                # Use consistent name
                pass  # Will be handled in prompt
        
        # Check for recurring phrases
        if japanese in context['recurring_phrases']:
            return context['recurring_phrases'][japanese]
        
        # Build context-aware prompt
        context_str = self._build_context_string(context)
        
        prompt = f"""Translate this Japanese manga text to English.

{context_str}

Strategy: {strategy.upper()}
Rules:
- Maintain consistency with previous translations
- Use established character names
- Keep it concise and natural

Japanese: "{japanese}"
English:"""
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        translation = response.text.strip().strip('"').strip()
        
        return translation
    
    def _build_context_string(self, context: Dict) -> str:
        """Build context string for prompt."""
        parts = []
        
        if context['character_names']:
            names = ', '.join([f"{jp}={en}" for jp, en in list(context['character_names'].items())[:5]])
            parts.append(f"Character names: {names}")
        
        if context['recurring_phrases']:
            phrases = list(context['recurring_phrases'].items())[:3]
            phrase_str = ', '.join([f'"{jp}"="{en}"' for jp, en in phrases])
            parts.append(f"Recurring phrases: {phrase_str}")
        
        if context['recent_strategies']:
            recent = context['recent_strategies'][-3:]
            parts.append(f"Recent page strategies: {', '.join(recent)}")
        
        return '\n'.join(parts) if parts else "Series context: First page"
