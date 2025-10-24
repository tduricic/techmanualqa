#!/usr/bin/env python3
"""
Retrieval Benchmark for TechManualQA Dataset

Evaluates different retrieval methods on finding the correct page given a question.
Tests both BM25 (lexical) and Dense retrieval (semantic) approaches.

Usage:
    uv run tech-retrieval-benchmark

    # Or run directly:
    python src/techmanualqa/retrieval_benchmark.py
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# For BM25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: rank_bm25 not found. Install with: uv add rank-bm25")
    sys.exit(1)

# For Dense Retrieval
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("ERROR: sentence-transformers not found. Install with: uv add sentence-transformers")
    sys.exit(1)

# For tokenization
try:
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
    print("WARNING: NLTK not found. Using simple split for tokenization.")
    print("Install with: uv add nltk")
    word_tokenize = None

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RetrievalBenchmark:
    """Benchmark retrieval methods on TechManualQA dataset."""

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
        print(f"Loading questions from {self.processed_dir}...")
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
                print(f"  WARNING: No candidates file found for {manual_name}")
                continue

            # Load questions from this manual
            manual_questions = []
            with open(candidates_file, 'r', encoding='utf-8') as f:
                for line in f:
                    question = json.loads(line)
                    manual_questions.append(question)

            print(f"  Loaded {len(manual_questions)} questions from {manual_name}")
            self.questions.extend(manual_questions)
            self.manual_dirs.append(manual_name)

        if not self.questions:
            raise ValueError(f"No questions found in {self.processed_dir}")

        print(f"\nTotal loaded: {len(self.questions)} questions from {len(self.manual_dirs)} manuals")

        # Filter out unanswerable questions
        total_before = len(self.questions)
        self.questions = [q for q in self.questions
                          if q.get('gt_page_number') != "None"
                          and q.get('category') != 'Unanswerable']

        unanswerable_count = total_before - len(self.questions)
        print(f"Filtered out {unanswerable_count} unanswerable questions")
        print(f"Final dataset: {len(self.questions)} answerable questions")

        # Count by category
        category_counts = Counter(q['category'] for q in self.questions)
        print(f"\nQuestions by category:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")

        # Group questions by manual
        self.questions_by_manual = defaultdict(list)
        for q in self.questions:
            doc_id = q['doc_id']
            # Extract manual name (remove .pdf extension)
            manual_name = doc_id.replace('.pdf', '')
            self.questions_by_manual[manual_name].append(q)

        # Cache for loaded pages
        self.manual_pages_cache = {}

    def load_manual_pages(self, manual_name: str) -> List[Dict]:
        """Load page content for a manual."""
        if manual_name in self.manual_pages_cache:
            return self.manual_pages_cache[manual_name]

        pages_file = self.processed_dir / manual_name / f"{manual_name}_pages.jsonl"

        if not pages_file.exists():
            print(f"WARNING: Pages file not found: {pages_file}")
            return []

        pages = []
        with open(pages_file, 'r', encoding='utf-8') as f:
            for line in f:
                page_data = json.loads(line)
                pages.append(page_data)

        self.manual_pages_cache[manual_name] = pages
        return pages

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        if word_tokenize:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        # Fallback to simple split
        return text.lower().split()

    def evaluate_bm25(self) -> Dict:
        """Evaluate BM25 retrieval."""
        print("\n" + "=" * 60)
        print("EVALUATING BM25 RETRIEVAL")
        print("=" * 60)

        results = []

        for manual_name, questions in tqdm(self.questions_by_manual.items(),
                                           desc="Processing manuals"):
            # Load pages for this manual
            pages = self.load_manual_pages(manual_name)

            if not pages:
                print(f"Skipping {manual_name} - no pages found")
                continue

            # Build BM25 index
            page_texts = [p['markdown_content'] for p in pages]
            page_numbers = [p['page_num'] for p in pages]

            # Tokenize corpus
            tokenized_corpus = [self.tokenize(text) for text in page_texts]

            # Initialize BM25
            bm25 = BM25Okapi(tokenized_corpus)

            # Evaluate each question
            for question in questions:
                query = question['question_text']
                gt_page = int(question['gt_page_number'])

                # Tokenize query
                tokenized_query = self.tokenize(query)

                # Get scores
                scores = bm25.get_scores(tokenized_query)

                # Rank pages by score (descending)
                ranked_indices = np.argsort(scores)[::-1]
                ranked_pages = [page_numbers[i] for i in ranked_indices]

                # Find rank of ground truth page (1-indexed)
                try:
                    gt_rank = ranked_pages.index(gt_page) + 1
                except ValueError:
                    gt_rank = len(ranked_pages) + 1  # Not found

                results.append({
                    'question_id': question['question_id'],
                    'manual': manual_name,
                    'category': question['category'],
                    'gt_page': gt_page,
                    'rank': gt_rank,
                    'method': 'BM25'
                })

        return self.compute_metrics(results)

    def evaluate_dense(self, model_name: str = 'BAAI/bge-small-en-v1.5') -> Dict:
        """
        Evaluate dense retrieval.

        Args:
            model_name: Hugging Face model name for embeddings
        """
        print("\n" + "=" * 60)
        print(f"EVALUATING DENSE RETRIEVAL ({model_name})")
        print("=" * 60)

        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        embedder = SentenceTransformer(model_name)

        results = []

        for manual_name, questions in tqdm(self.questions_by_manual.items(),
                                           desc="Processing manuals"):
            # Load pages for this manual
            pages = self.load_manual_pages(manual_name)

            if not pages:
                print(f"Skipping {manual_name} - no pages found")
                continue

            # Extract page content and numbers
            page_texts = [p['markdown_content'] for p in pages]
            page_numbers = [p['page_num'] for p in pages]

            # Encode all pages (batch for efficiency)
            print(f"  Encoding {len(page_texts)} pages for {manual_name}...")
            page_embeddings = embedder.encode(page_texts,
                                              convert_to_tensor=True,
                                              show_progress_bar=False)

            # Evaluate each question
            for question in questions:
                query = question['question_text']
                gt_page = int(question['gt_page_number'])

                # Encode query
                query_embedding = embedder.encode(query, convert_to_tensor=True)

                # Compute cosine similarities
                similarities = util.cos_sim(query_embedding, page_embeddings)[0]

                # Rank pages by similarity (descending)
                ranked_indices = similarities.argsort(descending=True).cpu().numpy()
                ranked_pages = [page_numbers[i] for i in ranked_indices]

                # Find rank of ground truth page
                try:
                    gt_rank = ranked_pages.index(gt_page) + 1
                except ValueError:
                    gt_rank = len(ranked_pages) + 1  # Not found

                results.append({
                    'question_id': question['question_id'],
                    'manual': manual_name,
                    'category': question['category'],
                    'gt_page': gt_page,
                    'rank': gt_rank,
                    'method': f'Dense-{model_name.split("/")[-1]}'
                })

        return self.compute_metrics(results)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute retrieval metrics from results."""
        df = pd.DataFrame(results)

        # Map specific categories to abstract categories
        def map_to_abstract_category(category: str) -> str:
            if category == "Procedural Step Inquiry":
                return "Procedural"
            else:
                # All other answerable categories are "General"
                return "General"

        df['abstract_category'] = df['category'].apply(map_to_abstract_category)

        # Overall metrics
        mrr_at_10 = self._compute_mrr(df['rank'], k=10)
        recall_at_5 = (df['rank'] <= 5).mean()
        recall_at_10 = (df['rank'] <= 10).mean()

        metrics = {
            'method': results[0]['method'],
            'total_questions': len(results),
            'MRR@10': mrr_at_10,
            'Recall@5': recall_at_5,
            'Recall@10': recall_at_10,
            'results_df': df
        }

        # Per-abstract-category metrics (General vs Procedural)
        category_metrics = {}
        for abstract_cat in ['General', 'Procedural']:
            cat_df = df[df['abstract_category'] == abstract_cat]
            if len(cat_df) > 0:
                category_metrics[abstract_cat] = {
                    'count': len(cat_df),
                    'MRR@10': self._compute_mrr(cat_df['rank'], k=10),
                    'Recall@5': (cat_df['rank'] <= 5).mean(),
                    'Recall@10': (cat_df['rank'] <= 10).mean()
                }

        metrics['by_category'] = category_metrics

        return metrics

    def _compute_mrr(self, ranks: pd.Series, k: int = 10) -> float:
        """Compute Mean Reciprocal Rank at k."""
        reciprocal_ranks = []
        for rank in ranks:
            if rank <= k:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks)

    def print_results(self, metrics: Dict):
        """Print metrics in a nice format."""
        print("\n" + "=" * 60)
        print(f"RESULTS: {metrics['method']}")
        print("=" * 60)
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"\nOverall Metrics:")
        print(f"  MRR@10:    {metrics['MRR@10']:.4f}")
        print(f"  Recall@5:  {metrics['Recall@5']:.4f} ({metrics['Recall@5'] * 100:.1f}%)")
        print(f"  Recall@10: {metrics['Recall@10']:.4f} ({metrics['Recall@10'] * 100:.1f}%)")

        if metrics['by_category']:
            print(f"\nBy Question Type:")
            print(f"{'Type':<15} {'Count':>6} {'MRR@10':>8} {'Recall@5':>9} {'Recall@10':>10}")
            print("-" * 60)

            # Print in specific order: General first, then Procedural
            for cat_name in ['General', 'Procedural']:
                if cat_name in metrics['by_category']:
                    cat_metrics = metrics['by_category'][cat_name]
                    print(f"{cat_name:<15} {cat_metrics['count']:>6} "
                          f"{cat_metrics['MRR@10']:>8.4f} "
                          f"{cat_metrics['Recall@5']:>9.1%} "
                          f"{cat_metrics['Recall@10']:>10.1%}")

    def save_results(self, all_metrics: List[Dict], output_dir: str = "paper_artifacts"):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Combine all results
        all_results = []
        for metrics in all_metrics:
            df = metrics['results_df'].copy()
            all_results.append(df)

        combined_df = pd.concat(all_results, ignore_index=True)

        # Save detailed results (includes both specific and abstract categories)
        results_file = output_path / 'retrieval_benchmark_detailed.csv'
        combined_df.to_csv(results_file, index=False)
        print(f"\n✓ Detailed results saved: {results_file}")

        # Save summary metrics
        summary = []
        for metrics in all_metrics:
            summary.append({
                'Method': metrics['method'],
                'Total_Questions': metrics['total_questions'],
                'MRR@10': round(metrics['MRR@10'], 4),
                'Recall@5': round(metrics['Recall@5'], 4),
                'Recall@10': round(metrics['Recall@10'], 4)
            })

        summary_df = pd.DataFrame(summary)
        summary_file = output_path / 'retrieval_benchmark_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Summary saved: {summary_file}")

        # Save by abstract category (General vs Procedural)
        category_comparison = []
        for metrics in all_metrics:
            for category, cat_metrics in metrics['by_category'].items():
                category_comparison.append({
                    'Method': metrics['method'],
                    'Question_Type': category,
                    'Count': cat_metrics['count'],
                    'MRR@10': round(cat_metrics['MRR@10'], 4),
                    'Recall@5': round(cat_metrics['Recall@5'], 4),
                    'Recall@10': round(cat_metrics['Recall@10'], 4)
                })

        if category_comparison:
            category_df = pd.DataFrame(category_comparison)
            category_file = output_path / 'retrieval_benchmark_by_type.csv'
            category_df.to_csv(category_file, index=False)
            print(f"✓ By question type results saved: {category_file}")


def main():
    """Main entry point for the retrieval benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run retrieval benchmark on TechManualQA dataset"
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed manuals with candidates (default: data/processed)"
    )
    parser.add_argument(
        "--dense-model",
        default="BAAI/bge-small-en-v1.5",
        help="Hugging Face model for dense retrieval (default: BAAI/bge-small-en-v1.5)"
    )
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts",
        help="Directory to save results (default: paper_artifacts)"
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 evaluation"
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense retrieval evaluation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TechManualQA Retrieval Benchmark")
    print("=" * 60)
    print(f"Processed data: {args.processed_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Initialize benchmark
    try:
        benchmark = RetrievalBenchmark(args.processed_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. You have processed the manuals (run tech-generate-candidates)")
        print("  2. Each manual folder has a *_candidates.jsonl file")
        print(f"  3. The processed directory exists at: {args.processed_dir}")
        sys.exit(1)

    all_metrics = []

    # Run BM25
    if not args.skip_bm25:
        start_time = time.time()
        bm25_metrics = benchmark.evaluate_bm25()
        bm25_time = time.time() - start_time

        benchmark.print_results(bm25_metrics)
        print(f"\nBM25 evaluation time: {bm25_time:.1f} seconds")

        all_metrics.append(bm25_metrics)

    # Run Dense Retrieval
    if not args.skip_dense:
        start_time = time.time()
        dense_metrics = benchmark.evaluate_dense(args.dense_model)
        dense_time = time.time() - start_time

        benchmark.print_results(dense_metrics)
        print(f"\nDense retrieval evaluation time: {dense_time:.1f} seconds")

        all_metrics.append(dense_metrics)

    # Save results
    if all_metrics:
        benchmark.save_results(all_metrics, args.output_dir)

    print("\n" + "=" * 60)
    print("RETRIEVAL BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()