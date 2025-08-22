import requests
import time
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


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

class AggressiveSummarizationStrategy(TokenReductionStrategy):
    """Reduces tokens by removing redundancy and condensing content"""
    
    def reduce_tokens(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove redundant phrases and filler words
        redundant_patterns = [
            r'\b(please|kindly|would you|could you)\b',
            r'\b(i think that|i believe that|in my opinion)\b',
            r'\b(obviously|clearly|of course)\b',
            r'\b(um|uh|well|so|like)\b(?=\s)',
            r'\b(actually|basically|essentially|really|quite|very|extremely)\b',
        ]
        
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Compress common phrases
        compressions = {
            r'as soon as possible': 'ASAP',
            r'for example': 'e.g.',
            r'that is to say': 'i.e.',
            r'in other words': 'i.e.',
            r'because of the fact that': 'because',
            r'in spite of the fact that': 'despite',
            r'in order to': 'to',
            r'due to the fact that': 'because',
            r'at this point in time': 'now',
        }
        
        for long_form, short_form in compressions.items():
            text = re.sub(long_form, short_form, text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def get_name(self) -> str:
        return "aggressive_summarization"

class KeywordExtractionStrategy(TokenReductionStrategy):
    """Reduces tokens by extracting key information only"""
    
    def reduce_tokens(self, text: str) -> str:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Keep sentences with important keywords
        important_keywords = [
            'data', 'analysis', 'result', 'conclusion', 'important', 'key', 'main',
            'problem', 'solution', 'error', 'issue', 'requirement', 'need', 'must',
            'create', 'build', 'develop', 'implement', 'generate', 'calculate',
            'explain', 'help', 'understand', 'learn', 'teach', 'show'
        ]
        
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Keep sentence if it contains important keywords or is very short
            if (any(keyword in sentence.lower() for keyword in important_keywords) or 
                len(sentence.split()) <= 8):
                filtered_sentences.append(sentence)
        
        return '. '.join(filtered_sentences)
    
    def get_name(self) -> str:
        return "keyword_extraction"

class StructuralCompressionStrategy(TokenReductionStrategy):
    """Reduces tokens by converting to structured format and removing fluff"""
    
    def reduce_tokens(self, text: str) -> str:
        # Split into sentences and compress
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and compress sentences
        compressed = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Remove common filler words
            fillers = ['actually', 'basically', 'essentially', 'really', 'quite', 
                      'very', 'extremely', 'obviously', 'clearly', 'definitely']
            
            for filler in fillers:
                sentence = re.sub(r'\b' + filler + r'\b', '', sentence, flags=re.IGNORECASE)
            
            # Clean up spaces
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            
            if sentence and len(sentence) > 5:
                compressed.append(sentence)
        
        # Join with concise separators
        return ' | '.join(compressed)
    
    def get_name(self) -> str:
        return "structural_compression"

class BulletPointStrategy(TokenReductionStrategy):
    """Converts verbose text to bullet points"""
    
    def reduce_tokens(self, text: str) -> str:
        sentences = re.split(r'[.!?]+', text)
        
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
            
            # Extract the core message by removing filler
            core = re.sub(r'\b(i would like|i need|could you|please|kindly)\b', '', sentence, flags=re.IGNORECASE)
            core = re.sub(r'\s+', ' ', core.strip())
            
            if core:
                key_points.append(f"• {core}")
        
        return '\n'.join(key_points)
    
    def get_name(self) -> str:
        return "bullet_points"

class LocalTokenCounter:
    """Simple token counter based on word approximation"""
    
    def count_tokens(self, text: str) -> int:
        # Rough approximation: 1 token ≈ 0.75 words for English
        words = len(text.split())
        return int(words * 1.33)  # Convert words to approximate tokens

class TokenGains:
    """Main class for managing token reduction and cost analysis with local LLMs"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b", 
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize TokenGains system for local models
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama server
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.token_counter = LocalTokenCounter()
        
        # Cost units (arbitrary units for comparison since local models are free)
        # This represents computational cost/time rather than monetary cost
        self.cost_per_token = 1.0  # 1 unit per token for comparison purposes
        
        self.strategies = [
            AggressiveSummarizationStrategy(),
            KeywordExtractionStrategy(),
            StructuralCompressionStrategy(),
            BulletPointStrategy()
        ]
        
        self.results: List[QueryResult] = []
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def query_local_model(self, prompt: str) -> Tuple[str, float]:
        """
        Query the local Ollama model
        
        Returns:
            Tuple of (response_text, latency_ms)
        """
        if not self.check_ollama_connection():
            raise ConnectionError(
                "Cannot connect to Ollama server. Make sure Ollama is running:\n"
                "1. Install Ollama from https://ollama.ai\n"
                "2. Run: ollama serve\n"
                "3. Pull a model: ollama pull llama3.2:3b"
            )
        
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200  # Limit response length
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            return result.get("response", ""), latency
            
        except requests.RequestException as e:
            raise Exception(f"Error querying model: {e}")
    
    def calculate_cost_units(self, tokens: int) -> float:
        """Calculate cost units based on token count"""
        return tokens * self.cost_per_token
    
    def evaluate_response_quality(self, original_response: str, reduced_response: str) -> float:
        """
        Evaluate response quality using semantic similarity,
        with length acting only as a penalty for extreme shrinkage/expansion.
        """
        if not original_response or not reduced_response:
            return 0.0

        # --- 1. Semantic similarity (core metric) ---
        try:
            vectorizer = TfidfVectorizer().fit([original_response, reduced_response])
            tfidf = vectorizer.transform([original_response, reduced_response])
            semantic_score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        except Exception:
            semantic_score = 0.0

        # --- 2. Structure preservation (sentence ratio) ---
        original_sentences = len(re.findall(r'[.!?]+', original_response))
        reduced_sentences = len(re.findall(r'[.!?]+', reduced_response))
        if original_sentences > 0:
            structure_score = min(reduced_sentences / original_sentences, 1.0)
        else:
            structure_score = 1.0

        # --- 3. Combine (semantic + structure) ---
        quality_score = (semantic_score * 0.8) + (structure_score * 0.2)

        # --- 4. Length penalty (only for extremes) ---
        length_ratio = len(reduced_response) / max(len(original_response), 1)
        if length_ratio < 0.1 or length_ratio > 1.2:
            quality_score *= 0.7  # penalize collapse/expansion

        return round(min(max(quality_score, 0.0), 1.0), 3)
    
    def optimize_query(self, prompt: str, strategy: TokenReductionStrategy) -> Tuple[str, int]:
        """Apply token reduction strategy to a prompt"""
        reduced_prompt = strategy.reduce_tokens(prompt)
        token_count = self.token_counter.count_tokens(reduced_prompt)
        return reduced_prompt, token_count
    
    def run_comparison(self, prompt: str) -> Dict:
        """
        Run comparison between original and optimized queries
        
        Args:
            prompt: The input prompt to optimize
            
        Returns:
            Dictionary with comparison results
        """
        print(f"🔍 Analyzing prompt: {prompt[:100]}...")
        
        original_tokens = self.token_counter.count_tokens(prompt)
        
        # Run original query
        print("  📤 Running original query...")
        try:
            original_response, original_latency = self.query_local_model(prompt)
            original_output_tokens = self.token_counter.count_tokens(original_response)
            original_cost = self.calculate_cost_units(original_tokens + original_output_tokens)
        except Exception as e:
            print(f"❌ Error with original query: {e}")
            return {"error": str(e)}
        
        # Test each strategy
        strategy_results = []
        
        for strategy in self.strategies:
            try:
                print(f"  🔧 Testing {strategy.get_name()}...")
                
                # Optimize prompt
                reduced_prompt, reduced_tokens = self.optimize_query(prompt, strategy)
                
                # Skip if no reduction achieved
                if reduced_tokens >= original_tokens:
                    print(f"    ⚠️  No token reduction achieved")
                    continue
                
                # Run optimized query
                reduced_response, reduced_latency = self.query_local_model(reduced_prompt)
                reduced_output_tokens = self.token_counter.count_tokens(reduced_response)
                reduced_cost = self.calculate_cost_units(reduced_tokens + reduced_output_tokens)
                
                # Calculate metrics
                token_reduction = ((original_tokens - reduced_tokens) / original_tokens) * 100
                cost_savings = ((original_cost - reduced_cost) / original_cost) * 100
                
                quality_score = self.evaluate_response_quality(original_response, reduced_response)
                
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
                
                print(f"    ✅ Tokens: {original_tokens} → {reduced_tokens} (-{token_reduction:.1f}%)")
                print(f"    💰 Cost units: {original_cost:.1f} → {reduced_cost:.1f} (-{cost_savings:.1f}%)")
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
            "strategy_breakdown": strategy_summary
        }

    def export_results(self, filename: str = "tokengains_results.json"):
        """Export results to JSON file for analysis"""
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
        
        print(f"📊 Results exported to {filename}")

def main():
    """Main function to run TokenGains with local model"""
    
    print("🚀 TokenGains - Local LLM Cost Optimization")
    print("=" * 50)
    
    # Initialize with local model
    print("🔧 Initializing TokenGains with local model...")
    
    try:
        token_gains = TokenGains(
            model_name="llama3.2:3b",  # You can change this to any model you have
            ollama_url="http://localhost:11434"
        )
        
        # Check connection
        if not token_gains.check_ollama_connection():
            print("❌ Cannot connect to Ollama server!")
            print("\n📋 Setup Instructions:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Start Ollama: ollama serve")
            print("3. Pull a model: ollama pull llama3.2:3b")
            print("4. Run this script again")
            return
        
        print("✅ Connected to Ollama server")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Test prompts
    test_prompts = [
        """
        I would really appreciate it if you could please help me understand how machine learning algorithms work. 
        I think that it's quite important for me to learn about this topic because I believe that it will be very 
        beneficial for my career. Could you kindly explain the basic concepts in a way that is easy to understand? 
        I'm particularly interested in supervised learning, unsupervised learning, and reinforcement learning.
        """,
        
        """
        I need to create a comprehensive business plan for my new startup company. The company will focus on 
        developing sustainable energy solutions for residential properties. I would like the plan to include 
        market analysis, competitive landscape, financial projections, marketing strategy, and operational 
        requirements. This is extremely important for securing funding from investors.
        """,
        
        """
        Can you please help me debug this Python code? I'm having issues with it and I can't figure out what's wrong. 
        The code is supposed to calculate the fibonacci sequence but it's not working properly. I think there might 
        be an error in the logic but I'm not sure where exactly the problem is occurring. Here's the code:
        def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)
        """
    ]
    
    # Run tests on each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 TEST {i}/{len(test_prompts)}")
        print("-" * 30)
        
        result = token_gains.run_comparison(prompt)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        if not result['strategies']:
            print("⚠️  No successful optimizations found")
            continue
        
        # Show best performing strategy
        best_strategy = max(result['strategies'], 
                          key=lambda x: x['metrics'].cost_savings_percent)
        
        print(f"\n🏆 Best Strategy: {best_strategy['strategy']}")
        metrics = best_strategy['metrics']
        print(f"   Token reduction: {metrics.token_reduction_percent:.1f}%")
        print(f"   Cost savings: {metrics.cost_savings_percent:.1f}%")
        print(f"   Quality retention: {metrics.response_quality_score:.2f}")
    
    # Print overall summary
    print(f"\n📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    summary = token_gains.get_performance_summary()
    
    if "error" not in summary:
        overall = summary["overall_metrics"]
        print(f"🤖 Model: {summary['model_used']}")
        print(f"🔢 Total runs: {summary['total_runs']}")
        print(f"📉 Average token reduction: {overall['avg_token_reduction']:.1f}%")
        print(f"💰 Average cost savings: {overall['avg_cost_savings']:.1f}%")
        print(f"🎯 Average quality retention: {overall['avg_quality_score']:.2f}")
        print(f"🚀 Maximum savings achieved: {overall['max_cost_savings']:.1f}%")
        
        print(f"\n📈 STRATEGY LEADERBOARD:")
        # Sort strategies by performance
        strategies = list(summary["strategy_breakdown"].items())
        strategies.sort(key=lambda x: x[1]['avg_cost_savings'], reverse=True)
        
        for i, (strategy, perf) in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
            print(f"     💰 Avg savings: {perf['avg_cost_savings']:.1f}%")
            print(f"     🎯 Avg quality: {perf['avg_quality_score']:.2f}")
            print(f"     📊 Runs: {perf['runs']}")
    
    # Export results
    token_gains.export_results()
    
    print(f"\n✅ TokenGains analysis complete!")

if __name__ == "__main__":
    main()