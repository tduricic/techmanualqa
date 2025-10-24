#!/usr/bin/env python3
"""
QA Benchmark for TechManualQA Dataset

Evaluates LLMs on generating answers to questions from technical manuals.
Tests three scenarios:
1. RAG: Retrieve top-k pages, then answer
2. Oracle: Provide only the ground truth page
3. Long Context: Provide the entire manual

Usage:
    uv run tech-qa-benchmark

    # Or run directly:
    python src/techmanualqa/qa_benchmark.py
"""

import json
import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict, Counter
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logging.debug("Loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables from .env won't be loaded automatically.")

# Import unified LLM client
try:
    from .llm_client import create_llm_client
except ImportError:
    from llm_client import create_llm_client

# For retrieval
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("ERROR: sentence-transformers not found. Install with: uv add sentence-transformers")
    sys.exit(1)

# For metrics
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("ERROR: rouge-score not found. Install with: uv add rouge-score")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def retry_with_exponential_backoff(
        max_retries: int = 5,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        retryable_errors: tuple = (Exception,)
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
        retryable_errors: Tuple of exception types to retry on

    Returns:
        Decorator function

    Raises:
        Last exception if all retries fail
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    error_msg = str(e)

                    # Check if it's a retryable error (503, 429, connection errors, etc.)
                    is_retryable = any(code in error_msg.lower() for code in [
                        '503', '429', '500', '502', '504',
                        'service unavailable', 'timeout', 'connection',
                        'rate limit', 'overloaded'
                    ])

                    if not is_retryable or attempt == max_retries - 1:
                        # Not retryable or final attempt - raise the exception
                        logging.error(f"Non-retryable error or max retries reached: {e}")
                        raise

                    # Calculate delay with exponential backoff
                    sleep_time = min(delay, max_delay)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed with error: {error_msg}. "
                        f"Retrying in {sleep_time:.1f} seconds..."
                    )
                    time.sleep(sleep_time)
                    delay *= exponential_base

            # Should not reach here, but raise last exception if we do
            raise last_exception

        return wrapper

    return decorator


class QABenchmark:
    """Benchmark LLM QA performance on TechManualQA dataset."""

    # Model configurations for LiteLLM
    LITELLM_MODELS = {
        'claude-4.5-sonnet': {'backend': 'litellm', 'model_name': 'bedrock-claude-4.5-sonnet', 'temperature': 0.0},
        'claude-4.5-haiku': {'backend': 'litellm', 'model_name': 'bedrock-claude-4.5-haiku', 'temperature': 0.0},
        'gpt4o': {'backend': 'litellm', 'model_name': 'azure-gpt4o', 'temperature': 0.0},
        'deepseek-v3': {'backend': 'litellm', 'model_name': 'deepseek-v3', 'temperature': 0.0},
        'o3-mini': {'backend': 'litellm', 'model_name': 'infobip-o3-mini', 'temperature': 1.0},
        'gpt-4-1': {'backend': 'litellm', 'model_name': 'infobip-gpt-4-1', 'temperature': 0.0},
    }

    def __init__(self, processed_dir: str = "data/processed"):
        """
        Initialize benchmark.

        Args:
            processed_dir: Directory containing processed manuals with candidates
        """
        self.processed_dir = Path(processed_dir)

        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found at {self.processed_dir}")

        # Load all questions from candidate files
        logging.info(f"Loading questions from {self.processed_dir}...")
        self.questions = []
        self.manual_dirs = []

        # Find all manual directories
        for manual_dir in sorted(self.processed_dir.iterdir()):
            if not manual_dir.is_dir():
                continue

            # Look for candidates file
            manual_name = manual_dir.name
            candidates_file = manual_dir / f"{manual_name}_candidates.jsonl"

            if not candidates_file.exists():
                logging.warning(f"  No candidates file found for {manual_name}")
                continue

            # Load questions from this manual
            manual_questions = []
            with open(candidates_file, 'r', encoding='utf-8') as f:
                for line in f:
                    question = json.loads(line)
                    # Filter out unanswerable questions
                    if question.get('category') != 'Unanswerable':
                        manual_questions.append(question)

            logging.info(f"  Loaded {len(manual_questions)} questions from {manual_name}")
            self.questions.extend(manual_questions)
            self.manual_dirs.append(manual_name)

        if not self.questions:
            raise ValueError(f"No questions found in {self.processed_dir}")

        logging.info(f"\nTotal loaded: {len(self.questions)} questions from {len(self.manual_dirs)} manuals")

        # Map categories to abstract types (excluding Unanswerable)
        def map_to_abstract_category(category: str) -> str:
            if category == "Procedural Step Inquiry":
                return "Procedural"
            else:
                return "General"

        for q in self.questions:
            q['abstract_category'] = map_to_abstract_category(q['category'])

        # Count by abstract category
        category_counts = Counter(q['abstract_category'] for q in self.questions)
        logging.info(f"\nQuestions by type:")
        for cat in ['General', 'Procedural']:
            count = category_counts.get(cat, 0)
            logging.info(f"  {cat}: {count}")

        # Group by manual
        self.questions_by_manual = defaultdict(list)
        for q in self.questions:
            doc_id = q['doc_id']
            manual_name = doc_id.replace('.pdf', '')
            self.questions_by_manual[manual_name].append(q)

        # Cache for pages and embeddings
        self.manual_pages_cache = {}
        self.manual_embeddings_cache = {}

        # Initialize retriever
        logging.info("\nLoading retriever model...")
        self.retriever = SentenceTransformer('BAAI/bge-small-en-v1.5')

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_manual_pages(self, manual_name: str) -> List[Dict]:
        """Load page content for a manual."""
        if manual_name in self.manual_pages_cache:
            return self.manual_pages_cache[manual_name]

        pages_file = self.processed_dir / manual_name / f"{manual_name}_pages.jsonl"

        if not pages_file.exists():
            logging.warning(f"Pages file not found: {pages_file}")
            return []

        pages = []
        with open(pages_file, 'r', encoding='utf-8') as f:
            for line in f:
                page_data = json.loads(line)
                pages.append(page_data)

        self.manual_pages_cache[manual_name] = pages
        return pages

    def get_page_embeddings(self, manual_name: str, pages: List[Dict]):
        """Get or compute embeddings for manual pages."""
        if manual_name in self.manual_embeddings_cache:
            return self.manual_embeddings_cache[manual_name]

        page_texts = [p['markdown_content'] for p in pages]
        embeddings = self.retriever.encode(page_texts,
                                           convert_to_tensor=True,
                                           show_progress_bar=False)

        self.manual_embeddings_cache[manual_name] = embeddings
        return embeddings

    def retrieve_pages(self, question_text: str, pages: List[Dict],
                       embeddings, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant pages for a question."""
        # Encode question
        query_embedding = self.retriever.encode(question_text, convert_to_tensor=True)

        # Compute similarities
        similarities = util.cos_sim(query_embedding, embeddings)[0]

        # Get top-k indices
        top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()

        # Return top pages
        return [pages[i] for i in top_indices]

    def build_context(self, pages: List[Dict], max_pages: Optional[int] = None) -> str:
        """Build context string from pages."""
        if max_pages:
            pages = pages[:max_pages]

        context_parts = []
        for page in pages:
            page_num = page.get('page_num', 'N/A')
            content = page.get('markdown_content', '')
            context_parts.append(f"--- Page {page_num} ---\n{content}")

        return "\n\n".join(context_parts)

    def build_qa_prompt(self, question: str, context: str) -> str:
        """Build prompt for QA task."""
        base_prompt = f"""You are an expert assistant helping users understand technical manuals.

Based on the following manual excerpt, answer the question accurately and concisely.

Manual excerpt:
{context}

Question: {question}

Answer: Provide a clear and accurate answer based only on the information in the excerpt."""

        return base_prompt

    def evaluate_answer(self, predicted: str, ground_truth: str,
                        abstract_category: str) -> Dict[str, float]:
        """
        Evaluate a predicted answer against ground truth.

        Returns dict with multiple metrics.
        """
        metrics = {}

        # Normalize text
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()

        # 1. Exact Match
        metrics['exact_match'] = 1.0 if pred_norm == gt_norm else 0.0

        # 2. Contains Match (is GT substring of prediction?)
        metrics['contains_match'] = 1.0 if gt_norm in pred_norm else 0.0

        # 3. ROUGE-L
        rouge_scores = self.rouge_scorer.score(ground_truth, predicted)
        metrics['rouge_l_f1'] = rouge_scores['rougeL'].fmeasure
        metrics['rouge_l_precision'] = rouge_scores['rougeL'].precision
        metrics['rouge_l_recall'] = rouge_scores['rougeL'].recall

        # 4. Token-level F1
        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            metrics['token_f1'] = 0.0
        else:
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(gt_tokens) if gt_tokens else 0

            if precision + recall == 0:
                metrics['token_f1'] = 0.0
            else:
                metrics['token_f1'] = 2 * (precision * recall) / (precision + recall)

        # 5. For Procedural: Step-level structure check
        if abstract_category == 'Procedural':
            procedural_words = ['step', 'first', 'then', 'next', 'finally', 'before', 'after']
            metrics['has_procedural_structure'] = 1.0 if any(w in pred_norm for w in procedural_words) else 0.0

        return metrics

    def generate_with_retry(self, llm, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate response with retry logic for handling server errors.

        Args:
            llm: LLM client instance
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """

        @retry_with_exponential_backoff(
            max_retries=5,
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            retryable_errors=(Exception,)
        )
        def _generate():
            return llm.generate(prompt, max_tokens=max_tokens)

        return _generate()

    def evaluate_scenario(self, model_name: str, scenario: str,
                          sample_size: Optional[int] = None,
                          top_k_retrieval: int = 3) -> Dict:
        """
        Evaluate a model on a specific scenario.

        Args:
            model_name: Name of model from LITELLM_MODELS
            scenario: 'rag', 'oracle', or 'long_context'
            sample_size: If set, only evaluate on this many questions (for testing)
            top_k_retrieval: For RAG, how many pages to retrieve

        Returns:
            Dict with results
        """
        if model_name not in self.LITELLM_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.LITELLM_MODELS.keys())}")

        logging.info(f"\n{'=' * 60}")
        logging.info(f"EVALUATING: {model_name} on {scenario.upper()} scenario")
        logging.info(f"{'=' * 60}")

        # Initialize LLM client
        model_config = self.LITELLM_MODELS[model_name]
        llm = create_llm_client(model_config)

        # Select questions to evaluate
        questions_to_eval = self.questions.copy()
        if sample_size:
            questions_to_eval = questions_to_eval[:sample_size]
            logging.info(f"Evaluating on sample of {sample_size} questions")

        results = []

        for question in tqdm(questions_to_eval, desc=f"{model_name} - {scenario}"):
            manual_name = question['doc_id'].replace('.pdf', '')
            pages = self.load_manual_pages(manual_name)

            if not pages:
                logging.warning(f"Skipping {question['question_id']} - no pages")
                continue

            # Build context based on scenario
            if scenario == 'oracle':
                # Use only the ground truth page
                gt_page_num = question.get('gt_page_number')
                if gt_page_num == "None":
                    # Skip questions without valid ground truth page
                    logging.warning(f"Skipping {question['question_id']} - no ground truth page")
                    continue

                gt_page_num = int(gt_page_num)
                context_pages = [p for p in pages if p['page_num'] == gt_page_num]
                context = self.build_context(context_pages)

            elif scenario == 'rag':
                # Retrieve top-k pages
                embeddings = self.get_page_embeddings(manual_name, pages)
                context_pages = self.retrieve_pages(question['question_text'],
                                                    pages, embeddings,
                                                    top_k=top_k_retrieval)
                context = self.build_context(context_pages)

            elif scenario == 'long_context':
                # Use entire manual (limit to avoid token limits)
                context = self.build_context(pages, max_pages=100)

            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            # Build prompt and generate answer WITH RETRY LOGIC
            prompt = self.build_qa_prompt(question['question_text'], context)

            try:
                # Use the new generate_with_retry method
                predicted_answer = self.generate_with_retry(llm, prompt, max_tokens=500)
                generation_success = True
            except Exception as e:
                logging.error(f"Error generating for {question['question_id']} after all retries: {e}")
                predicted_answer = ""
                generation_success = False

            # Evaluate
            if generation_success and predicted_answer:
                metrics = self.evaluate_answer(
                    predicted_answer,
                    question['gt_answer_snippet'],
                    question['abstract_category']
                )
            else:
                # Failed generation - all zeros
                metrics = {
                    'exact_match': 0.0,
                    'contains_match': 0.0,
                    'rouge_l_f1': 0.0,
                    'rouge_l_precision': 0.0,
                    'rouge_l_recall': 0.0,
                    'token_f1': 0.0
                }

            # Collect result
            result = {
                'question_id': question['question_id'],
                'doc_id': question['doc_id'],
                'model': model_name,
                'scenario': scenario,
                'category': question['category'],
                'abstract_category': question['abstract_category'],
                'question_text': question['question_text'],
                'gt_answer': question['gt_answer_snippet'],
                'predicted_answer': predicted_answer,
                'generation_success': generation_success,
                **metrics
            }

            results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Compute aggregate metrics
        metrics = self.compute_aggregate_metrics(results_df, model_name, scenario)
        metrics['results_df'] = results_df

        return metrics

    def compute_aggregate_metrics(self, results_df: pd.DataFrame,
                                  model_name: str, scenario: str) -> Dict:
        """Compute aggregate metrics from results."""
        metrics = {
            'model': model_name,
            'scenario': scenario,
            'total_questions': len(results_df)
        }

        # Overall metrics
        metrics['generation_success_rate'] = results_df['generation_success'].mean()

        # Only compute metrics on successful generations
        successful_df = results_df[results_df['generation_success']]

        if len(successful_df) > 0:
            metrics['exact_match'] = successful_df['exact_match'].mean()
            metrics['contains_match'] = successful_df['contains_match'].mean()
            metrics['rouge_l_f1'] = successful_df['rouge_l_f1'].mean()
            metrics['token_f1'] = successful_df['token_f1'].mean()
        else:
            metrics['exact_match'] = 0.0
            metrics['contains_match'] = 0.0
            metrics['rouge_l_f1'] = 0.0
            metrics['token_f1'] = 0.0

        # Breakdown by abstract category (General and Procedural only)
        category_metrics = {}
        for abstract_cat in ['General', 'Procedural']:
            cat_df = successful_df[successful_df['abstract_category'] == abstract_cat]

            if len(cat_df) > 0:
                category_metrics[abstract_cat] = {
                    'count': len(cat_df),
                    'exact_match': cat_df['exact_match'].mean(),
                    'rouge_l_f1': cat_df['rouge_l_f1'].mean(),
                    'token_f1': cat_df['token_f1'].mean()
                }

        metrics['by_category'] = category_metrics

        return metrics

    def print_results(self, metrics: Dict):
        """Print results in readable format."""
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {metrics['model']} - {metrics['scenario'].upper()}")
        print(f"{'=' * 60}")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Generation Success Rate: {metrics['generation_success_rate']:.1%}")

        print(f"\nOverall Metrics:")
        print(f"  Exact Match:    {metrics['exact_match']:.4f} ({metrics['exact_match'] * 100:.1f}%)")
        print(f"  Contains Match: {metrics['contains_match']:.4f} ({metrics['contains_match'] * 100:.1f}%)")
        print(f"  ROUGE-L F1:     {metrics['rouge_l_f1']:.4f}")
        print(f"  Token F1:       {metrics['token_f1']:.4f}")

        print(f"\nBy Question Type:")
        print(f"{'Type':<15} {'Count':>6} {'Exact':>7} {'ROUGE-L':>8} {'Token-F1':>8}")
        print("-" * 60)

        for cat_name in ['General', 'Procedural']:
            if cat_name in metrics['by_category']:
                cat_metrics = metrics['by_category'][cat_name]
                print(f"{cat_name:<15} {cat_metrics['count']:>6} "
                      f"{cat_metrics['exact_match']:>7.1%} "
                      f"{cat_metrics['rouge_l_f1']:>8.4f} "
                      f"{cat_metrics['token_f1']:>8.4f}")

    def save_results(self, all_metrics: List[Dict], output_dir: str = "paper_artifacts"):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Combine all detailed results
        all_results = []
        for metrics in all_metrics:
            df = metrics['results_df'].copy()
            all_results.append(df)

        combined_df = pd.concat(all_results, ignore_index=True)

        # Save detailed results
        detailed_file = output_path / 'qa_benchmark_detailed.csv'
        combined_df.to_csv(detailed_file, index=False)
        logging.info(f"\n✓ Detailed results saved: {detailed_file}")

        # Save summary
        summary = []
        for metrics in all_metrics:
            summary.append({
                'Model': metrics['model'],
                'Scenario': metrics['scenario'],
                'Total_Questions': metrics['total_questions'],
                'Success_Rate': round(metrics['generation_success_rate'], 4),
                'Exact_Match': round(metrics['exact_match'], 4),
                'Contains_Match': round(metrics['contains_match'], 4),
                'ROUGE_L_F1': round(metrics['rouge_l_f1'], 4),
                'Token_F1': round(metrics['token_f1'], 4)
            })

        summary_df = pd.DataFrame(summary)
        summary_file = output_path / 'qa_benchmark_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"✓ Summary saved: {summary_file}")

        # Save by question type (General and Procedural only)
        type_comparison = []
        for metrics in all_metrics:
            for cat_type, cat_metrics in metrics['by_category'].items():
                row = {
                    'Model': metrics['model'],
                    'Scenario': metrics['scenario'],
                    'Question_Type': cat_type,
                    'Count': cat_metrics['count'],
                    'Exact_Match': round(cat_metrics['exact_match'], 4),
                    'ROUGE_L_F1': round(cat_metrics['rouge_l_f1'], 4),
                    'Token_F1': round(cat_metrics['token_f1'], 4)
                }
                type_comparison.append(row)

        type_df = pd.DataFrame(type_comparison)
        type_file = output_path / 'qa_benchmark_by_type.csv'
        type_df.to_csv(type_file, index=False)
        logging.info(f"✓ By question type results saved: {type_file}")


def main():
    """Main entry point for the QA benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run QA benchmark on TechManualQA dataset"
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed manuals with candidates (default: data/processed)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=['gpt-4-1'],
        help="Models to evaluate (default: gpt-4-1)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=['oracle', 'rag'],
        choices=['oracle', 'rag', 'long_context'],
        help="Scenarios to evaluate (default: oracle rag)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="If set, only evaluate on this many questions (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts",
        help="Directory to save results (default: paper_artifacts)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of pages to retrieve for RAG scenario (default: 3)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TechManualQA QA Benchmark")
    print("=" * 60)
    print(f"Processed data: {args.processed_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Check environment variables
    if not os.getenv("LITELLM_API_KEY"):
        print("\nERROR: LITELLM_API_KEY not set in environment")
        print("Set it in your .env file or with: export LITELLM_API_KEY=your_key")
        sys.exit(1)

    # Initialize benchmark
    try:
        benchmark = QABenchmark(args.processed_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. You have generated candidates (run tech-generate-candidates)")
        print("  2. Each manual folder has a *_candidates.jsonl file")
        print(f"  3. The processed directory exists at: {args.processed_dir}")
        sys.exit(1)

    all_metrics = []

    # Run evaluations
    for model_name in args.models:
        for scenario in args.scenarios:
            try:
                start_time = time.time()

                metrics = benchmark.evaluate_scenario(
                    model_name=model_name,
                    scenario=scenario,
                    sample_size=args.sample_size,
                    top_k_retrieval=args.top_k
                )

                elapsed = time.time() - start_time

                benchmark.print_results(metrics)
                print(f"\nTime: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")

                all_metrics.append(metrics)

            except Exception as e:
                logging.error(f"Failed to evaluate {model_name} on {scenario}: {e}", exc_info=True)
                continue

    # Save all results
    if all_metrics:
        benchmark.save_results(all_metrics, args.output_dir)

    print(f"\n{'=' * 60}")
    print("QA BENCHMARK COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()