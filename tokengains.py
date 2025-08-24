import requests
import time
import json
import re
import os
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Optional: sentence embeddings & HF tokenizer
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from transformers import AutoTokenizer  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False


@dataclass
class QueryResult:
    """Results from a single query execution"""
    original_tokens: int
    reduced_tokens: int
    token_reduction_percent: float
    original_cost_units: float
    reduced_cost_units: float
    cost_savings_percent: float
    response_quality_score: float
    latency_ms: float
    strategy_used: str

class TokenReductionStrategy(ABC):
    """Abstract base class for token reduction strategies"""
    
    @abstractmethod
    def reduce_tokens(self, text: str) -> str:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class ImprovedAggressiveStrategy(TokenReductionStrategy):
    """Enhanced aggressive summarization with better regex patterns"""
    
    def reduce_tokens(self, text: str) -> str:
        # Remove excessive whitespace first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Enhanced redundant patterns with more comprehensive coverage
        redundant_patterns = [
            # Courtesy language
            r'\b(please|kindly|would you|could you|can you|will you)\b',
            r'\b(i would appreciate|i would be grateful|thank you)\b',
            
            # Opinion markers
            r'\b(i think that|i believe that|in my opinion|from my perspective)\b',
            r'\b(it seems to me|i feel that|i suspect that)\b',
            
            # Obviousness markers
            r'\b(obviously|clearly|of course|naturally|certainly)\b',
            r'\b(without a doubt|it goes without saying)\b',
            
            # Filler words and hesitation
            r'\b(um|uh|well|so|like|you know|i mean)\b(?=\s)',
            r'\b(sort of|kind of|more or less)\b',
            
            # Intensifiers (often unnecessary)
            r'\b(actually|basically|essentially|really|quite|very|extremely)\b',
            r'\b(absolutely|completely|totally|entirely|utterly)\b',
            r'\b(significantly|considerably|substantially)\b',
            
            # Redundant time references
            r'\b(currently|at present|at this moment|right now)\b',
            r'\b(in the future|going forward|moving forward)\b',
        ]
        
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Enhanced phrase compressions
        compressions = {
            # Time expressions
            r'as soon as possible': 'ASAP',
            r'at this point in time': 'now',
            r'in the near future': 'soon',
            r'at the present time': 'now',
            
            # Examples and clarifications
            r'for example': 'e.g.',
            r'for instance': 'e.g.',
            r'that is to say': 'i.e.',
            r'in other words': 'i.e.',
            r'such as': 'like',
            
            # Causal relationships
            r'because of the fact that': 'because',
            r'due to the fact that': 'because',
            r'in spite of the fact that': 'despite',
            r'despite the fact that': 'despite',
            r'in order to': 'to',
            r'for the purpose of': 'to',
            r'with the intention of': 'to',
            
            # Common verbose phrases
            r'a large number of': 'many',
            r'a small number of': 'few',
            r'make a decision': 'decide',
            r'come to a conclusion': 'conclude',
            r'give consideration to': 'consider',
            r'make an assumption': 'assume',
            r'conduct an analysis': 'analyze',
            r'perform a review': 'review',
        }
        
        for long_form, short_form in compressions.items():
            text = re.sub(long_form, short_form, text, flags=re.IGNORECASE)
        
        # Remove redundant articles and prepositions where safe
        text = re.sub(r'\b(the|a|an)\s+(same|following|above|below|previous|next)\b', r'\2', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\s*,\s*,\s*', ', ', text)  # Fix double commas
        text = re.sub(r'\s*\.\s*\.\s*', '. ', text)  # Fix double periods
        
        return text
    
    def get_name(self) -> str:
        return "improved_aggressive"

class DomainAwareStrategy(TokenReductionStrategy):
    """Domain-specific compression based on content analysis"""
    
    def reduce_tokens(self, text: str) -> str:
        text_lower = text.lower()
        
        # Detect domain and apply specific strategies
        if self._is_code_domain(text_lower):
            return self._compress_code_request(text)
        elif self._is_business_domain(text_lower):
            return self._compress_business_request(text)
        elif self._is_academic_domain(text_lower):
            return self._compress_academic_request(text)
        elif self._is_creative_domain(text_lower):
            return self._compress_creative_request(text)
        else:
            return self._generic_compression(text)
    
    def _is_code_domain(self, text: str) -> bool:
        code_indicators = ['code', 'debug', 'python', 'javascript', 'java', 'function', 
                          'variable', 'class', 'method', 'algorithm', 'programming', 'syntax']
        return sum(1 for word in code_indicators if word in text) >= 2
    
    def _is_business_domain(self, text: str) -> bool:
        business_indicators = ['business', 'marketing', 'strategy', 'revenue', 'profit', 
                              'customer', 'market', 'investment', 'budget', 'plan']
        return sum(1 for word in business_indicators if word in text) >= 2
    
    def _is_academic_domain(self, text: str) -> bool:
        academic_indicators = ['research', 'study', 'analysis', 'theory', 'hypothesis', 
                              'methodology', 'conclusion', 'literature', 'academic']
        return sum(1 for word in academic_indicators if word in text) >= 2
    
    def _is_creative_domain(self, text: str) -> bool:
        creative_indicators = ['story', 'creative', 'write', 'character', 'plot', 
                              'design', 'artistic', 'narrative', 'poem']
        return sum(1 for word in creative_indicators if word in text) >= 2
    
    def _compress_code_request(self, text: str) -> str:
        # Preserve technical terms, be very direct
        essential_code_patterns = [
            r'(debug|fix|error|issue|problem)',
            r'(function|method|class|variable)',
            r'(python|javascript|java|c\+\+|html|css)',
            r'(code|script|program)',
        ]
        
        sentences = re.split(r'[.!?]+', text)
        compressed = []
        
        for sent in sentences:
            sent = sent.strip()
            if any(re.search(pattern, sent, re.IGNORECASE) for pattern in essential_code_patterns):
                # Remove courtesy language but keep technical details
                clean = re.sub(r'\b(please|could you|would you|i need|help me)\b', '', sent, flags=re.IGNORECASE)
                clean = re.sub(r'\b(i think|i believe|i\'m having)\b', '', clean, flags=re.IGNORECASE)
                clean = re.sub(r'\s+', ' ', clean.strip())
                if clean:
                    compressed.append(clean)
        
        result = '. '.join(compressed)
        return result if result else "Debug code issue."
    
    def _compress_business_request(self, text: str) -> str:
        # Focus on deliverables and requirements
        key_business_terms = ['plan', 'strategy', 'analysis', 'budget', 'revenue', 
                             'market', 'customer', 'investment', 'roi']
        
        sentences = re.split(r'[.!?]+', text)
        essential_info = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if (any(term in sent_lower for term in key_business_terms) or
                'include' in sent_lower or 'need' in sent_lower or 'require' in sent_lower):
                
                # Compress business jargon
                clean = sent
                clean = re.sub(r'\b(comprehensive|detailed|thorough)\b', '', clean, flags=re.IGNORECASE)
                clean = re.sub(r'\b(i would like|i need|it would be helpful)\b', 'Need:', clean, flags=re.IGNORECASE)
                clean = re.sub(r'\s+', ' ', clean.strip())
                if clean:
                    essential_info.append(clean)
        
        return '. '.join(essential_info)
    
    def _compress_academic_request(self, text: str) -> str:
        # Preserve methodology and key concepts
        academic_markers = ['explain', 'analyze', 'compare', 'evaluate', 'discuss', 
                           'research', 'study', 'theory', 'concept']
        
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []
        
        for sent in sentences:
            if any(marker in sent.lower() for marker in academic_markers):
                # Remove academic fluff but keep precision
                clean = re.sub(r'\b(it would be beneficial|i would appreciate)\b', '', sent, flags=re.IGNORECASE)
                clean = re.sub(r'\b(comprehensive understanding of|detailed explanation of)\b', '', clean, flags=re.IGNORECASE)
                clean = re.sub(r'\s+', ' ', clean.strip())
                if clean:
                    key_sentences.append(clean)
        
        return '. '.join(key_sentences)
    
    def _compress_creative_request(self, text: str) -> str:
        # Preserve creative requirements and constraints
        creative_elements = ['story', 'character', 'plot', 'setting', 'theme', 
                            'style', 'tone', 'genre', 'creative']
        
        sentences = re.split(r'[.!?]+', text)
        creative_specs = []
        
        for sent in sentences:
            if any(element in sent.lower() for element in creative_elements):
                clean = re.sub(r'\b(i would love|i would like|it would be great)\b', 'Create:', sent, flags=re.IGNORECASE)
                clean = re.sub(r'\s+', ' ', clean.strip())
                if clean:
                    creative_specs.append(clean)
        
        return '. '.join(creative_specs)
    
    def _generic_compression(self, text: str) -> str:
        return ImprovedAggressiveStrategy().reduce_tokens(text)
    
    def get_name(self) -> str:
        return "domain_aware"

class LengthAwareStrategy(TokenReductionStrategy):
    """Adaptive compression based on text length with better quality preservation"""
    
    def reduce_tokens(self, text: str) -> str:
        word_count = len(text.split())
        
        if word_count <= 30:
            # Minimal compression for short texts
            return self._minimal_compression(text)
        elif word_count <= 80:
            # Light compression for medium texts
            return self._light_compression(text)
        elif word_count <= 150:
            # Medium compression
            return self._medium_compression(text)
        else:
            # Aggressive compression for long texts
            return self._aggressive_compression(text)
    
    def _minimal_compression(self, text: str) -> str:
        # Only remove the most obvious redundancies
        text = re.sub(r'\b(obviously|clearly|of course)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _light_compression(self, text: str) -> str:
        # Remove filler words but preserve structure
        fillers = ['actually', 'basically', 'really', 'quite', 'very']
        for filler in fillers:
            text = re.sub(r'\b' + filler + r'\b', '', text, flags=re.IGNORECASE)
        
        # Simple phrase replacements
        text = re.sub(r'in order to', 'to', text, flags=re.IGNORECASE)
        text = re.sub(r'because of the fact that', 'because', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _medium_compression(self, text: str) -> str:
        # More aggressive removal while preserving meaning
        patterns = [
            r'\b(please|kindly|would you|could you)\b',
            r'\b(i think that|i believe that|in my opinion)\b',
            r'\b(actually|basically|essentially|really|quite|very)\b',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Phrase compressions
        compressions = {
            r'for example': 'e.g.',
            r'that is to say': 'i.e.',
            r'as soon as possible': 'ASAP',
            r'at this point in time': 'now',
        }
        
        for long_form, short_form in compressions.items():
            text = re.sub(long_form, short_form, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _aggressive_compression(self, text: str) -> str:
        return ImprovedAggressiveStrategy().reduce_tokens(text)
    
    def get_name(self) -> str:
        return "length_aware"

class KeywordExtractionStrategy(TokenReductionStrategy):
    """Enhanced keyword extraction with domain-specific keywords"""
    
    def reduce_tokens(self, text: str) -> str:
        # Expanded keyword list with domain-specific terms
        important_keywords = [
            # Action words
            'create', 'build', 'develop', 'implement', 'generate', 'calculate', 'design',
            'explain', 'help', 'understand', 'learn', 'teach', 'show', 'demonstrate',
            'analyze', 'review', 'evaluate', 'assess', 'compare', 'optimize',
            
            # Problem/solution words
            'problem', 'solution', 'error', 'issue', 'bug', 'fix', 'debug',
            'requirement', 'need', 'must', 'should', 'important', 'critical',
            
            # Domain-specific terms
            'data', 'analysis', 'result', 'conclusion', 'research', 'study',
            'code', 'function', 'algorithm', 'method', 'python', 'javascript',
            'business', 'marketing', 'strategy', 'plan', 'budget', 'revenue',
            'model', 'system', 'process', 'workflow', 'automation',
        ]
        
        sentences = re.split(r'[.!?]+', text)
        filtered_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            
            # Keep sentence if it contains important keywords, is short, or has numbers/specifics
            if (any(keyword in sentence_lower for keyword in important_keywords) or 
                len(sentence.split()) <= 6 or
                re.search(r'\d+', sentence) or  # Contains numbers
                re.search(r'[A-Z]{2,}', sentence)):  # Contains acronyms
                
                filtered_sentences.append(sentence)
        
        return '. '.join(filtered_sentences)
    
    def get_name(self) -> str:
        return "enhanced_keyword_extraction"

class StructuralCompressionStrategy(TokenReductionStrategy):
    """Enhanced structural compression with better separators"""
    
    def reduce_tokens(self, text: str) -> str:
        sentences = re.split(r'[.!?]+', text)
        
        compressed = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 8:  # Skip very short sentences
                continue
            
            # Remove common filler words more comprehensively
            fillers = ['actually', 'basically', 'essentially', 'really', 'quite', 
                      'very', 'extremely', 'obviously', 'clearly', 'definitely',
                      'absolutely', 'completely', 'totally', 'certainly']
            
            for filler in fillers:
                sentence = re.sub(r'\b' + filler + r'\b', '', sentence, flags=re.IGNORECASE)
            
            # Remove redundant phrases
            sentence = re.sub(r'\b(i would like|i need|could you|please help me)\b', '', sentence, flags=re.IGNORECASE)
            
            # Clean up spaces
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            
            if sentence and len(sentence) > 5:
                compressed.append(sentence)
        
        # Use more concise separators based on content
        if len(compressed) <= 3:
            return '. '.join(compressed)  # Keep periods for short lists
        else:
            return ' | '.join(compressed)  # Use pipes for longer lists
    
    def get_name(self) -> str:
        return "enhanced_structural"

class BulletPointStrategy(TokenReductionStrategy):
    """Enhanced bullet point conversion with better filtering"""
    
    def reduce_tokens(self, text: str) -> str:
        sentences = re.split(r'[.!?]+', text)
        
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4:  # Skip very short sentences
                continue
            
            # Extract core message more aggressively
            core = sentence
            
            # Remove courtesy language
            courtesy_patterns = [
                r'\b(i would like|i would appreciate|i need|could you|please|kindly)\b',
                r'\b(would you mind|it would be helpful|i hope)\b',
                r'\b(thank you|thanks|gratefully)\b'
            ]
            
            for pattern in courtesy_patterns:
                core = re.sub(pattern, '', core, flags=re.IGNORECASE)
            
            # Remove opinion markers
            core = re.sub(r'\b(i think|i believe|i feel|in my opinion)\b', '', core, flags=re.IGNORECASE)
            
            # Clean up
            core = re.sub(r'\s+', ' ', core.strip())
            
            if core and len(core) > 10:
                # Make it more action-oriented
                if not re.match(r'^(create|build|analyze|explain|help|show)', core, re.IGNORECASE):
                    if any(word in core.lower() for word in ['explain', 'understand', 'learn']):
                        core = f"Explain: {core}"
                    elif any(word in core.lower() for word in ['create', 'build', 'develop']):
                        core = f"Create: {core}"
                    elif any(word in core.lower() for word in ['analyze', 'review', 'evaluate']):
                        core = f"Analyze: {core}"
                
                key_points.append(f"• {core}")
        
        return '\n'.join(key_points) if key_points else "• " + text.strip()
    
    def get_name(self) -> str:
        return "enhanced_bullet_points"

class EnhancedTokenCounter:
    """More accurate token counting with caching"""
    
    def __init__(self, hf_tokenizer=None, calibration_factor: float = 1.0):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.hf_tokenizer = hf_tokenizer
        self.calibration_factor = calibration_factor
    
    def count_tokens(self, text: str) -> int:
        # Use text hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.cache:
            self.cache_hits += 1
            return self.cache[text_hash]
        
        self.cache_misses += 1
        
        # Prefer HF tokenizer if provided
        if self.hf_tokenizer is not None:
            try:
                token_ids = self.hf_tokenizer(text, add_special_tokens=False).input_ids
                token_count = len(token_ids)
            except Exception:
                token_count = None
        else:
            token_count = None

        if token_count is None:
            # Heuristic token counting using multiple methods
            words = len(text.split())
            chars = len(text)
            word_estimate = int(words * 1.33)  # 1 token ≈ 0.75 words
            char_estimate = chars // 4         # ~3.5-4 chars per token
            special_chars = len(re.findall(r'[^\w\s]', text))
            punctuation_tokens = special_chars // 2
            token_count = int((word_estimate * 0.5 + char_estimate * 0.4 + punctuation_tokens * 0.1))

        # Apply calibration factor
        token_count = max(0, int(round(token_count * self.calibration_factor)))
        
        # Cache the result
        self.cache[text_hash] = token_count
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
        
        return token_count
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache),
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }

class TokenGains:
    """Enhanced TokenGains with quick wins implemented"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b", 
                 ollama_url: str = "http://localhost:11434",
                 hf_tokenizer_name: Optional[str] = None,
                 calibration_factor: float = 1.0,
                 min_quality_threshold: float = 0.60,
                 cache_file: str = "response_cache.jsonl",
                 provider: str = "local",
                 openai_api_key: Optional[str] = None):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.min_quality_threshold = min_quality_threshold
        self.cache_file = cache_file
        self.provider = provider
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Optional HF tokenizer for more accurate token counts
        hf_tok = None
        if hf_tokenizer_name and _HAS_HF:
            try:
                hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name, use_fast=True)
            except Exception:
                hf_tok = None

        self.token_counter = EnhancedTokenCounter(hf_tokenizer=hf_tok, calibration_factor=calibration_factor)
        self.cost_per_token = 1.0
        
        # Enhanced strategies with quick wins
        self.strategies = [
            ImprovedAggressiveStrategy(),
            DomainAwareStrategy(),
            LengthAwareStrategy(),
            KeywordExtractionStrategy(),
            StructuralCompressionStrategy(),
            BulletPointStrategy(),
            # New constraint-preserving strategy
            ConstraintPreservingStrategy()
        ]
        
        self.results: List[QueryResult] = []
        self.response_cache = {}  # Cache for model responses (in-memory)
        self._load_response_cache()  # Populate from disk if available

        # Lazy sentence-embedding model
        self._st_model = None

    def _load_response_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("model") == self.model_name and "key" in rec and "response" in rec:
                                self.response_cache[rec["key"]] = (rec["response"], rec.get("latency_ms", 0.0))
                        except Exception:
                            continue
        except Exception:
            pass

    def _append_response_cache(self, key: str, response: str, latency_ms: float):
        try:
            with open(self.cache_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "model": self.model_name,
                    "key": key,
                    "response": response,
                    "latency_ms": latency_ms,
                }) + "\n")
        except Exception:
            pass
    
    def check_connection(self) -> bool:
        """Check if the selected provider is available"""
        if self.provider == "local":
            return self.check_ollama_connection()
        elif self.provider == "hosted":
            return self.check_openai_connection()
        return False
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def check_openai_connection(self) -> bool:
        """Check if OpenAI API key is valid"""
        if not self.openai_api_key:
            return False
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def query_local_model(self, prompt: str) -> Tuple[str, float]:
        """Query with caching, retries, and backoff"""
        # Create cache key
        cache_key = hashlib.md5(f"{self.provider}:{self.model_name}:{prompt}".encode()).hexdigest()
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        if self.provider == "local":
            return self._query_ollama(prompt, cache_key)
        elif self.provider == "hosted":
            return self._query_openai(prompt, cache_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _query_ollama(self, prompt: str, cache_key: str) -> Tuple[str, float]:
        """Query Ollama local model"""
        if not self.check_ollama_connection():
            raise ConnectionError(
                "Cannot connect to Ollama server. Make sure Ollama is running:\n"
                "1. Install Ollama from https://ollama.ai\n"
                "2. Run: ollama serve\n"
                f"3. Pull the model: ollama pull {self.model_name}"
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200
            }
        }

        attempts = 0
        last_err: Optional[Exception] = None
        while attempts < 3:
            attempts += 1
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                latency = (time.time() - start_time) * 1000
                response_data = (result.get("response", ""), latency)

                # Cache in memory and persist to disk
                self.response_cache[cache_key] = response_data
                self._append_response_cache(cache_key, response_data[0], response_data[1])

                # Limit cache size
                if len(self.response_cache) > 200:
                    oldest_keys = list(self.response_cache.keys())[:40]
                    for key in oldest_keys:
                        try:
                            del self.response_cache[key]
                        except KeyError:
                            pass
                return response_data
            except requests.RequestException as e:
                last_err = e
                # Exponential backoff
                time.sleep(min(2 ** attempts, 8))
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempts, 8))
        raise Exception(f"Error querying model after {attempts} attempts: {last_err}")
    
    def _query_openai(self, prompt: str, cache_key: str) -> Tuple[str, float]:
        """Query OpenAI API"""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required for hosted mode. Set OPENAI_API_KEY environment variable or pass --api-key"
            )

        # Map common model names to OpenAI equivalents
        openai_model = self.model_name
        if self.model_name.startswith("llama") or self.model_name.startswith("deepseek"):
            openai_model = "gpt-4o-mini"  # Default fallback

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "model": openai_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }

        attempts = 0


        last_err: Optional[Exception] = None
        while attempts < 3:
            attempts += 1
            start_time = time.time()
            # Add delay for hosted provider
            time.sleep(10)  # 10 second delay between requests

            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                latency = (time.time() - start_time) * 1000
                
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                response_data = (content, latency)

                # Cache the response
                self.response_cache[cache_key] = response_data
                self._append_response_cache(cache_key, response_data[0], response_data[1])

                return response_data
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempts, 8))
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempts, 8))
        
        raise Exception(f"Error querying OpenAI after {attempts} attempts: {last_err}")
    
    def calculate_cost_units(self, tokens: int) -> float:
        """Calculate cost units based on token count"""
        return tokens * self.cost_per_token
    
    def evaluate_response_quality(self, original_response: str, reduced_response: str) -> float:
        """Enhanced quality evaluation with sentence embeddings + TF-IDF fallback and length-aware scoring"""
        if not original_response or not reduced_response:
            return 0.0

        # Semantic similarity (primary metric)
        semantic_score = 0.0
        # Try sentence embeddings
        if _HAS_ST:
            try:
                if self._st_model is None:
                    self._st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                emb = self._st_model.encode([original_response, reduced_response], normalize_embeddings=True)
                # Cosine since normalized
                semantic_score = float((emb[0] @ emb[1]).item() if hasattr(emb[0], 'item') else emb[0].dot(emb[1]))
            except Exception:
                semantic_score = 0.0
        if semantic_score == 0.0:
            # Fallback to TF-IDF
            try:
                vectorizer = TfidfVectorizer().fit([original_response, reduced_response])
                tfidf = vectorizer.transform([original_response, reduced_response])
                semantic_score = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
            except Exception:
                semantic_score = 0.0

        # Structure preservation
        original_sentences = len(re.findall(r'[.!?]+', original_response))
        reduced_sentences = len(re.findall(r'[.!?]+', reduced_response))
        
        if original_sentences > 0:
            structure_score = min(reduced_sentences / original_sentences, 1.0)
        else:
            structure_score = 1.0

        # Key concept retention (enhanced)
        original_concepts = set(re.findall(r'\b[A-Za-z]{4,}\b', original_response.lower()))
        reduced_concepts = set(re.findall(r'\b[A-Za-z]{4,}\b', reduced_response.lower()))
        
        if original_concepts:
            concept_retention = len(original_concepts & reduced_concepts) / len(original_concepts)
        else:
            concept_retention = 1.0

        # Combine scores with improved weights
        quality_score = (semantic_score * 0.5 + 
                        structure_score * 0.2 + 
                        concept_retention * 0.3)

        # Length penalty (more nuanced)
        length_ratio = len(reduced_response) / max(len(original_response), 1)
        
        if length_ratio < 0.05:  # Too aggressive compression
            quality_score *= 0.5
        elif length_ratio < 0.15:  # Very aggressive but acceptable
            quality_score *= 0.8
        elif length_ratio > 1.1:  # Response got longer (bad)
            quality_score *= 0.9

        return round(min(max(quality_score, 0.0), 1.0), 3)
    
    def select_best_strategies(self, prompt: str) -> List[TokenReductionStrategy]:
        """Smart strategy selection based on prompt analysis"""
        text_lower = prompt.lower()
        word_count = len(prompt.split())
        
        selected_strategies = []
        
        # Always include the improved aggressive strategy
        selected_strategies.append(ImprovedAggressiveStrategy())
        
        # Add domain-aware if we can detect domain
        domain_indicators = {
            'code': ['code', 'debug', 'function', 'python', 'javascript'],
            'business': ['business', 'plan', 'strategy', 'marketing'],
            'academic': ['research', 'study', 'analysis', 'explain'],
        }
        
        for domain, indicators in domain_indicators.items():
            if sum(1 for indicator in indicators if indicator in text_lower) >= 2:
                selected_strategies.append(DomainAwareStrategy())
                break
        
        # Add length-aware for longer texts
        if word_count > 50:
            selected_strategies.append(LengthAwareStrategy())
        
        # Add keyword extraction for information-rich texts
        if word_count > 30:
            selected_strategies.append(KeywordExtractionStrategy())
        
        # Add structural for very long texts
        if word_count > 100:
            selected_strategies.append(StructuralCompressionStrategy())

        # Add constraint-preserving when constraints or code clues exist
        if any(tok in text_lower for tok in ["```", "traceback", "error", "exception", "must", "should", "shall"]):
            selected_strategies.append(ConstraintPreservingStrategy())
        
        # Limit to top 4 strategies to avoid over-testing
        return selected_strategies[:4]
    
    def run_comparison(self, prompt: str) -> Dict:
        """Enhanced comparison with smart strategy selection and early-exit evaluation"""
        print(f"🔍 Analyzing prompt: {prompt[:100]}...")
        
        original_tokens = self.token_counter.count_tokens(prompt)
        
        # Smart strategy selection
        selected_strategies = self.select_best_strategies(prompt)
        print(f"  🎯 Selected {len(selected_strategies)} strategies for testing")
        
        # Run original query
        print("  📤 Running original query...")
        try:
            original_response, original_latency = self.query_local_model(prompt)
            original_output_tokens = self.token_counter.count_tokens(original_response)
            original_cost = self.calculate_cost_units(original_tokens + original_output_tokens)
                
        except Exception as e:
            print(f"❌ Error with original query: {e}")
            return {"error": str(e)}
        
        # Phase 1: fast reductions in parallel to choose top candidates (by token cut)
        preliminary: List[Tuple[TokenReductionStrategy, str, int]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(selected_strategies))) as ex:
            futures = {ex.submit(self.optimize_query, prompt, s): s for s in selected_strategies}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    reduced_prompt, reduced_tokens = fut.result()
                    preliminary.append((s, reduced_prompt, reduced_tokens))
                except Exception as e:
                    print(f"    ❌ Reduction failed for {s.get_name()}: {e}")
                    continue

        # Filter by minimum 5% input token reduction
        prelim_filtered = [p for p in preliminary if p[2] < original_tokens * 0.95]
        if not prelim_filtered:
            return {
                "original_prompt": prompt,
                "original_tokens": original_tokens,
                "original_cost": original_cost,
                "strategies": []
            }

        # Rank by estimated savings (input token cut) and take top 2
        prelim_filtered.sort(key=lambda t: (original_tokens - t[2]) / max(1, original_tokens), reverse=True)
        top_candidates = prelim_filtered[:2]

        # Phase 2: query model for the top candidates
        strategy_results: List[Dict] = []
        for strategy, reduced_prompt, reduced_tokens in top_candidates:
            try:
                print(f"  🔧 Testing {strategy.get_name()}...")

                reduced_response, reduced_latency = self.query_local_model(reduced_prompt)
                reduced_output_tokens = self.token_counter.count_tokens(reduced_response)
                reduced_cost = self.calculate_cost_units(reduced_tokens + reduced_output_tokens)

                token_reduction = ((original_tokens - reduced_tokens) / original_tokens) * 100
                cost_savings = ((original_cost - reduced_cost) / original_cost) * 100

                quality_score = self.evaluate_response_quality(original_response, reduced_response)

                # Enforce minimum quality threshold
                if quality_score < self.min_quality_threshold:
                    print(f"    ⚠️  Discarded due to low quality ({quality_score:.2f} < {self.min_quality_threshold:.2f})")
                    continue

                result = QueryResult(
                    original_tokens=original_tokens,
                    reduced_tokens=reduced_tokens,
                    token_reduction_percent=token_reduction,
                    original_cost_units=original_cost,
                    reduced_cost_units=reduced_cost,
                    cost_savings_percent=cost_savings,
                    response_quality_score=quality_score,
                    latency_ms=reduced_latency,
                    strategy_used=strategy.get_name()
                )

                strategy_results.append({
                    "strategy": strategy.get_name(),
                    "reduced_prompt": reduced_prompt,
                    "original_response": original_response,
                    "reduced_response": reduced_response,
                    "metrics": result
                })

                self.results.append(result)

                token_trend = "↓" if reduced_tokens < original_tokens else ("↑" if reduced_tokens > original_tokens else "→")
                print(f"    ✅ Tokens: {original_tokens} → {reduced_tokens} ({token_trend} {abs(token_reduction):.1f}%)")

                cost_trend = "↓" if reduced_cost < original_cost else ("↑" if reduced_cost > original_cost else "→")
                print(f"    💰 Cost units: {original_cost:.1f} → {reduced_cost:.1f} ({cost_trend} {abs(cost_savings):.1f}%)")
                print(f"    🎯 Quality: {quality_score:.2f}")

            except Exception as e:
                print(f"    ❌ Error with strategy {strategy.get_name()}: {e}")
                continue
        
        return {
            "original_prompt": prompt,
            "original_tokens": original_tokens,
            "original_cost": original_cost,
            "strategies": strategy_results
        }
    
    def optimize_query(self, prompt: str, strategy: TokenReductionStrategy) -> Tuple[str, int]:
        """Apply token reduction strategy to a prompt"""
        reduced_prompt = strategy.reduce_tokens(prompt)
        token_count = self.token_counter.count_tokens(reduced_prompt)
        return reduced_prompt, token_count
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary across all runs"""
        if not self.results:
            return {"error": "No results available"}
        
        # Aggregate statistics
        token_reductions = [r.token_reduction_percent for r in self.results]
        cost_savings = [r.cost_savings_percent for r in self.results]
        quality_scores = [r.response_quality_score for r in self.results]
        
        # Group by strategy
        strategy_performance = {}
        for result in self.results:
            strategy = result.strategy_used
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "token_reductions": [],
                    "cost_savings": [],
                    "quality_scores": []
                }
            
            strategy_performance[strategy]["token_reductions"].append(result.token_reduction_percent)
            strategy_performance[strategy]["cost_savings"].append(result.cost_savings_percent)
            strategy_performance[strategy]["quality_scores"].append(result.response_quality_score)
        
        # Calculate averages for each strategy
        strategy_summary = {}
        for strategy, data in strategy_performance.items():
            strategy_summary[strategy] = {
                "avg_token_reduction": statistics.mean(data["token_reductions"]),
                "avg_cost_savings": statistics.mean(data["cost_savings"]),
                "avg_quality_score": statistics.mean(data["quality_scores"]),
                "runs": len(data["token_reductions"])
            }
        
        # Add cache statistics
        cache_stats = self.token_counter.get_cache_stats()
        
        return {
            "total_runs": len(self.results),
            "model_used": self.model_name,
            "overall_metrics": {
                "avg_token_reduction": statistics.mean(token_reductions),
                "avg_cost_savings": statistics.mean(cost_savings),
                "avg_quality_score": statistics.mean(quality_scores),
                "max_token_reduction": max(token_reductions),
                "max_cost_savings": max(cost_savings),
                "best_quality": max(quality_scores)
            },
            "strategy_breakdown": strategy_summary,
            "cache_performance": cache_stats
        }

    def export_results(self, filename: str = "enhanced_tokengains_results.json"):
        """Export enhanced results to JSON file"""
        results_data = {
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self.get_performance_summary(),
            "detailed_results": [
                {
                    "strategy": r.strategy_used,
                    "original_tokens": r.original_tokens,
                    "reduced_tokens": r.reduced_tokens,
                    "token_reduction_percent": r.token_reduction_percent,
                    "cost_savings_percent": r.cost_savings_percent,
                    "quality_score": r.response_quality_score,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📊 Enhanced results exported to {filename}")

    def analyze_prompt_characteristics(self, prompt: str) -> Dict:
        """Analyze prompt to provide insights for optimization"""
        analysis = {
            "word_count": len(prompt.split()),
            "char_count": len(prompt),
            "sentence_count": len(re.findall(r'[.!?]+', prompt)),
            "avg_words_per_sentence": len(prompt.split()) / max(len(re.findall(r'[.!?]+', prompt)), 1),
        }
        
        # Detect redundancy patterns
        redundancy_patterns = {
            "courtesy_language": len(re.findall(r'\b(please|kindly|would you|could you)\b', prompt, re.IGNORECASE)),
            "filler_words": len(re.findall(r'\b(actually|basically|really|very|quite)\b', prompt, re.IGNORECASE)),
            "opinion_markers": len(re.findall(r'\b(i think|i believe|in my opinion)\b', prompt, re.IGNORECASE)),
            "repetitive_phrases": len(re.findall(r'\b(\w+)\s+\1\b', prompt, re.IGNORECASE))
        }
        
        analysis["redundancy_score"] = sum(redundancy_patterns.values())
        analysis["redundancy_breakdown"] = redundancy_patterns
        
        # Suggest optimal strategies
        if analysis["word_count"] > 100:
            analysis["recommended_strategies"] = ["improved_aggressive", "domain_aware", "length_aware"]
        elif analysis["redundancy_score"] > 5:
            analysis["recommended_strategies"] = ["improved_aggressive", "enhanced_structural"]
        else:
            analysis["recommended_strategies"] = ["domain_aware", "enhanced_keyword_extraction"]
        
        return analysis


class ConstraintPreservingStrategy(TokenReductionStrategy):
    """Preserve constraints, code fences, stack traces, steps, and critical requirements while trimming fluff."""

    def reduce_tokens(self, text: str) -> str:
        # Capture preserved blocks
        preserved_blocks = []

        # Code fences ``` ... ```
        code_fence_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        def _store(match):
            block = match.group(0).strip()
            preserved_blocks.append(block)
            return ""
        text_wo_code = code_fence_pattern.sub(_store, text)

        # Inline code `...`
        inline_code_pattern = re.compile(r"`[^`\n]+`")
        def _store_inline(m):
            preserved_blocks.append(m.group(0))
            return ""
        text_wo_code = inline_code_pattern.sub(_store_inline, text_wo_code)

        # Stack traces / errors
        lines = text_wo_code.splitlines()
        kept_lines = []
        for ln in lines:
            l = ln.strip()
            if not l:
                continue
            if (
                l.startswith("Traceback") or
                re.search(r"\b(File \".*?\", line \d+)\b", l) or
                re.search(r"\bError|Exception|at\b", l, re.IGNORECASE)
            ):
                preserved_blocks.append(ln)
                continue
            # Numbered or bulleted steps
            if re.match(r"^\s*(\d+\.|-\s+|\*)\s+", ln):
                kept_lines.append(ln)
                continue
            # Quoted text
            if re.search(r'"[^"]+"|\'[^\']+\'', ln):
                kept_lines.append(ln)
                continue
            # Requirements: must/should/shall, constraints with numbers+units
            if re.search(r"\b(must|should|shall|required)\b", l, re.IGNORECASE) or re.search(r"\b\d+(?:\.\d+)?\s?(ms|s|sec|msec|kb|mb|gb|tb|%|px)\b", l, re.IGNORECASE):
                kept_lines.append(ln)
                continue
            # For code-domain clues keep function signatures or snippets-like lines
            if re.search(r"\bdef\s+\w+\(|\bclass\s+\w+\(|\w+\s*=\s*lambda|;|\{\}|=>", l):
                kept_lines.append(ln)
                continue

        # Clean remaining text aggressively for pleasantries
        remaining_text = " ".join([ln for ln in lines if ln not in preserved_blocks])
        remaining_text = re.sub(r"\b(please|kindly|would you|could you|i would like|i think|i believe|thanks|thank you)\b", "", remaining_text, flags=re.IGNORECASE)
        remaining_text = re.sub(r"\s+", " ", remaining_text).strip()

        # Build concise output
        parts = []
        if kept_lines:
            parts.append(" | ".join([re.sub(r"\s+", " ", k.strip()) for k in kept_lines]))
        if remaining_text:
            parts.append(remaining_text)
        # Append code/inline code blocks at the end to preserve fidelity
        parts.extend(preserved_blocks)
        return "\n".join([p for p in parts if p])

    def get_name(self) -> str:
        return "constraint_preserving"

def main():
    """Enhanced main function with better testing and analysis"""
    
    # Add argument parser
    parser = argparse.ArgumentParser(description='Enhanced TokenGains - Local LLM Cost Optimization')
    parser.add_argument('--model', '-m', 
                       default='llama3.2:3b',
                       help='Model name to use (default: llama3.2:3b for local, gpt-4o-mini for hosted)')
    parser.add_argument('--url', '-u',
                       default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--provider', '-p',
                       choices=['local', 'hosted'],
                       default='local',
                       help='Provider to use: local (Ollama) or hosted (OpenAI)')
    parser.add_argument('--api-key',
                       help='OpenAI API key (can also use OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Adjust default model based on provider
    if args.provider == 'hosted' and args.model == 'llama3.2:3b':
        args.model = 'gpt-4o-mini'
    
    print("🚀 Enhanced TokenGains - Local LLM Cost Optimization")
    print("=" * 60)
    
    # Initialize enhanced TokenGains with CLI arguments
    print(f"🔧 Initializing Enhanced TokenGains")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model}")
    if args.provider == 'local':
        print(f"   Ollama URL: {args.url}")
    
    try:
        token_gains = TokenGains(
            model_name=args.model,
            ollama_url=args.url,
            provider=args.provider,
            openai_api_key=args.api_key
        )
        
        # Check connection
        if not token_gains.check_connection():
            if args.provider == 'local':
                print("❌ Cannot connect to Ollama server!")
                print("\n📋 Setup Instructions:")
                print("1. Install Ollama: https://ollama.ai")
                print("2. Start Ollama: ollama serve")
                print(f"3. Pull the model: ollama pull {args.model}")
                print("4. Run this script again")
            else:
                print("❌ Cannot connect to OpenAI API!")
                print("\n📋 Setup Instructions:")
                print("1. Get an API key from https://platform.openai.com/api-keys")
                print("2. Set environment variable: set OPENAI_API_KEY=your-key-here")
                print("3. Or use --api-key parameter")
                print("4. Run this script again")
            return
        
        print("✅ Connected successfully")
        print(f"🎯 Using model: {args.model}")
        print(f"🎯 Loaded {len(token_gains.strategies)} enhanced strategies")

        # Output preview controls for original vs optimized model outputs (disabled in favor of file export)
        SHOW_OUTPUT_PREVIEWS = False
        OUTPUT_PREVIEW_CHARS = 400
        # File to store full best-strategy pairs per test (JSONL)
        BEST_PAIR_FILE = "best_strategy_pairs.jsonl"
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Enhanced test prompts with various characteristics
    test_prompts = [
        # Verbose courtesy language (high redundancy)
        """
        I would really appreciate it if you could please help me understand how machine learning algorithms work. 
        I think that it's quite important for me to learn about this topic because I believe that it will be very 
        beneficial for my career. Could you kindly explain the basic concepts in a way that is easy to understand? 
        I'm particularly interested in supervised learning, unsupervised learning, and reinforcement learning.
        """,
        
        # Business request with jargon (medium complexity)
        """
        I need to create a comprehensive business plan for my new startup company. The company will focus on 
        developing sustainable energy solutions for residential properties. I would like the plan to include 
        market analysis, competitive landscape, financial projections, marketing strategy, and operational 
        requirements. This is extremely important for securing funding from investors.
        """,
        
        # Code debugging request (technical domain)
        """
        Can you please help me debug this Python code? I'm having issues with it and I can't figure out what's wrong. 
        The code is supposed to calculate the fibonacci sequence but it's not working properly. I think there might 
        be an error in the logic but I'm not sure where exactly the problem is occurring. Here's the code:
        def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)
        """,
        
        # Long academic request (high complexity)
        """
        I would greatly appreciate a comprehensive explanation of quantum computing principles and their practical applications.
        I believe that understanding quantum superposition, quantum entanglement, and quantum interference is very important
        for my research. Could you please provide detailed information about how these concepts work together in quantum algorithms?
        Additionally, I think it would be extremely helpful if you could explain the current limitations of quantum computing
        technology and discuss the potential timeline for widespread commercial adoption in various industries such as cryptography,
        drug discovery, and financial modeling. I would be very grateful for any insights you could share.
        """,
        
        # Concise technical request (low redundancy)
        """
        Optimize pandas DataFrame operations for 1M+ rows. Need efficient groupby and rolling window calculations.
        Focus on memory usage and parallel processing strategies.
        """
    ]
    
    # Analyze and test each prompt
    total_original_cost = 0
    total_optimized_cost = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 TEST {i}/{len(test_prompts)}")
        print("-" * 40)
        
        # Analyze prompt characteristics
        analysis = token_gains.analyze_prompt_characteristics(prompt)
        print(f"📊 Prompt Analysis:")
        print(f"   Words: {analysis['word_count']}, Sentences: {analysis['sentence_count']}")
        print(f"   Redundancy Score: {analysis['redundancy_score']}")
        print(f"   Recommended: {', '.join(analysis['recommended_strategies'])}")
        
        # Run comparison
        result = token_gains.run_comparison(prompt)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        total_original_cost += result['original_cost']
        
        if not result['strategies']:
            print("⚠️  No successful optimizations found")
            continue
        
        # Show top 3 performing strategies
        sorted_strategies = sorted(result['strategies'], 
                                 key=lambda x: x['metrics'].cost_savings_percent, 
                                 reverse=True)
        
        print(f"\n🏆 TOP STRATEGIES:")
        for j, strategy in enumerate(sorted_strategies[:3], 1):
            metrics = strategy['metrics']
            # Only count the best (top-1) strategy toward total optimized cost per test
            if j == 1:
                total_optimized_cost += metrics.reduced_cost_units
            
            print(f"   {j}. {strategy['strategy']}")
            print(f"      💰 Cost savings: {metrics.cost_savings_percent:.1f}%")
            print(f"      📉 Token reduction: {metrics.token_reduction_percent:.1f}%")
            print(f"      🎯 Quality: {metrics.response_quality_score:.2f}")
            
            if j == 1:  # Persist the best strategy pair for offline inspection
                print(f"      📝 Optimized prompt: {strategy['reduced_prompt'][:100]}...")
                try:
                    record = {
                        "test_index": i,
                        "model": token_gains.model_name,
                        "strategy": strategy['strategy'],
                        "original_prompt": prompt,
                        "optimized_prompt": strategy['reduced_prompt'],
                        "original_response": strategy.get('original_response') or "",
                        "optimized_response": strategy.get('reduced_response') or "",
                        "metrics": {
                            "token_reduction_percent": metrics.token_reduction_percent,
                            "cost_savings_percent": metrics.cost_savings_percent,
                            "quality_score": metrics.response_quality_score,
                            "original_tokens": metrics.original_tokens,
                            "reduced_tokens": metrics.reduced_tokens,
                            "original_cost_units": metrics.original_cost_units,
                            "reduced_cost_units": metrics.reduced_cost_units,
                            "latency_ms": metrics.latency_ms,
                        },
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(BEST_PAIR_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as _e:
                    print(f"      ⚠️  Could not write best pair file: {_e}")
    
    # Enhanced final summary
    print(f"\n📊 ENHANCED PERFORMANCE SUMMARY")
    print("=" * 60)
    
    summary = token_gains.get_performance_summary()
    
    if "error" not in summary:
        overall = summary["overall_metrics"]
        cache_stats = summary["cache_performance"]
        
        print(f"🤖 Model: {summary['model_used']}")
        print(f"🔢 Total runs: {summary['total_runs']}")
        print(f"📉 Average token reduction: {overall['avg_token_reduction']:.1f}%")
        print(f"💰 Average cost savings: {overall['avg_cost_savings']:.1f}%")
        print(f"🎯 Average quality retention: {overall['avg_quality_score']:.2f}")
        print(f"🚀 Maximum savings achieved: {overall['max_cost_savings']:.1f}%")
        
        # Show total cost impact
        if total_original_cost > 0:
            total_savings_percent = ((total_original_cost - total_optimized_cost) / total_original_cost) * 100
            print(f"💡 Total cost reduction across all tests: {total_savings_percent:.1f}%")
        
        # Cache performance
        print(f"\n⚡ Cache Performance:")
        print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Cache size: {cache_stats['cache_size']} entries")
        
        print(f"\n📈 ENHANCED STRATEGY LEADERBOARD:")
        strategies = list(summary["strategy_breakdown"].items())
        strategies.sort(key=lambda x: x[1]['avg_cost_savings'], reverse=True)
        
        for i, (strategy, perf) in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
            print(f"     💰 Avg savings: {perf['avg_cost_savings']:.1f}%")
            print(f"     🎯 Avg quality: {perf['avg_quality_score']:.2f}")
            print(f"     📊 Runs: {perf['runs']}")
    
    # Export enhanced results
    token_gains.export_results()
    
  

if __name__ == "__main__":
    main()
