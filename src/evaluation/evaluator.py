import os
import json
import time
import argparse
import logging
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
    handlers=[logging.FileHandler("eval_pipeline.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("RAGEval")


# schema
class LLMJudgeResult(BaseModel):
    faithfulness_score: int = Field(
        description="Điểm từ 0 đến 10 đánh giá việc mô hình bám sát Context."
    )
    faithfulness_reasoning: str = Field(description="Lý do ngắn gọn cho điểm faithfulness.")
    correctness_score: int = Field(
        description="Điểm từ 0 đến 10 đánh giá độ chính xác của câu trả lời so với Ground Truth."
    )
    correctness_reasoning: str = Field(description="Lý do ngắn gọn cho điểm correctness.")


# rag api client
class RAGClient:
    def __init__(self, api_url: str, session_id: str):
        self.api_url = api_url
        self.session_id = session_id

    def query(self, question: str) -> Tuple[List[str], str]:
        url = f"{self.api_url}?query={question}&session_id={self.session_id}"
        full_answer = ""
        retrieved_contexts = []

        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    logger.error(f"API Error: {r.status_code} for query: {question[:30]}...")
                    return [], ""

                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if chunk.get("type") == "sources":
                                for doc in chunk.get("data", []):
                                    if "content" in doc:
                                        retrieved_contexts.append(doc["content"])
                            elif chunk.get("type") == "content":
                                full_answer += chunk.get("data", "")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"RAG Request Exception: {e}")

        return retrieved_contexts, full_answer


# semantic evaluator
class SemanticEvaluator:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", threshold=0.2):
        logger.info(f"Loading Semantic Evaluator Model: {model_name}...")
        self.matcher = CrossEncoder(model_name, max_length=512, device="cuda")
        self.threshold = threshold

    # compute retrieval metrics
    def compute_metrics(
        self, ground_truths: List[str], retrieved_chunks: List[str], top_k=5
    ) -> Dict[str, float]:
        if not retrieved_chunks or not ground_truths:
            return {"hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "match_ranks": []}

        eval_chunks = retrieved_chunks[:top_k]
        match_ranks = []
        gt_matched_count = 0

        for gt in ground_truths:
            pairs = [[gt, chunk] for chunk in eval_chunks]
            scores = self.matcher.predict(pairs)

            for rank, score in enumerate(scores):
                if score >= self.threshold:
                    match_ranks.append(rank + 1)
                    gt_matched_count += 1
                    break

        hit_at_k = 10.0 if match_ranks else 0.0
        mrr = (1.0 / min(match_ranks) * 10.0) if match_ranks else 0.0
        recall_at_k = (gt_matched_count / len(ground_truths)) * 10.0

        return {
            "hit_at_k": hit_at_k,
            "mrr": mrr,
            "recall_at_k": recall_at_k,
            "match_ranks": match_ranks,
        }


# llm judge evaluator
class LLMEvaluator:
    def __init__(self, max_retries=5, base_delay=2.0):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")
        self.client = genai.Client(api_key=api_key)
        self.judge_model = "gemini-2.5-flash"
        self.max_retries = max_retries
        self.base_delay = base_delay

    # evaluate single question
    def evaluate(
        self, question: str, q_type: str, context: str, gen_answer: str, gt_answer: str
    ) -> Dict[str, Any]:
        prompt = f"""
        Bạn là một hệ thống đánh giá AI. Hãy chấm điểm câu trả lời của RAG system dựa trên các thông tin sau.
        
        [DỮ LIỆU ĐẦU VÀO]
        - Câu hỏi: {question}
        - Loại câu hỏi: {q_type}
        - Ground Truth Answer: {gt_answer}
        - Retrieved Context: {context}
        - Generated Answer: {gen_answer}

        [HƯỚNG DẪN CHẤM ĐIỂM (0 - 10)]
        1. Faithfulness (Tính trung thực): Generated Answer có bám sát 100% vào Retrieved Context không? 
           - Bịa đặt thông tin (hallucination) -> Trừ điểm nặng (0-3).
           - Nếu Generated Answer báo "Không có thông tin" và Context thực sự KHÔNG có thông tin -> 10 điểm.
        2. Correctness (Tính chính xác): Generated Answer giải quyết đúng trọng tâm của Ground Truth Answer không?
           - Bỏ qua khác biệt về từ vựng, chỉ xét ngữ nghĩa.
           - Nếu q_type là 'unanswerable' và Generated Answer từ chối trả lời hợp lý -> Correctness = 10.
        """

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.judge_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LLMJudgeResult,
                        temperature=0.0,
                    ),
                )
                return json.loads(response.text)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_delay * (2**attempt)
                    logger.warning(
                        f"API Error (Attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"LLM Judge FAILED after {self.max_retries} attempts: {e}")
                    return {"faithfulness_score": 0, "correctness_score": 0, "error": str(e)}


# main pipeline
# bulk evaluation pipeline
class EvaluationPipeline:
    def __init__(self, dataset_path: str, output_path: str, api_url: str, session_id: str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.checkpoint_path = output_path.replace(".csv", "_checkpoint.jsonl")
        self.rag_client = RAGClient(api_url, session_id)
        self.semantic_evaluator = SemanticEvaluator()
        self.llm_evaluator = LLMEvaluator()

    def _categorize_error(
        self, q_type: str, hit_at_k: float, faithfulness: int, correctness: int
    ) -> str:
        """categorize root cause error"""
        if q_type == "unanswerable":
            return (
                "Success"
                if correctness >= 8
                else "Generation Failure (Hallucination on Unanswerable)"
            )

        if hit_at_k == 0.0:
            return "Retrieval Failure (Context not found in Top-K)"

        if correctness >= 8:
            return "Success"

        # gray area (partial success)
        if 5 <= correctness <= 7:
            if faithfulness >= 8:
                return "Partial Success (Incomplete Context)"
            return "Partial Success (Generation is noisy/vague)"

        # absolute or generation failure
        if correctness < 5:
            if faithfulness >= 8:
                return "Context Failure (Matched but useless/truncated)"
            return "Generation Failure (Failed to extract or reasoned poorly)"

        return "Unknown"

    def _load_checkpoint(self) -> List[Dict]:
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                logger.info("Checkpoint found. Resuming evaluation from last saved state...")
                return [json.loads(line) for line in f]
        return []

    def run(self, top_k: int = 5):
        logger.info(f"Starting Evaluation Pipeline on {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        results = self._load_checkpoint()
        processed_count = len(results)
        remaining_dataset = dataset[processed_count:]

        if not remaining_dataset:
            logger.info("done!")
            self._generate_report(results)
            return

        with open(self.checkpoint_path, "a", encoding="utf-8") as checkpoint_file:
            for item in tqdm(
                remaining_dataset, desc=f"Evaluating (Resuming from {processed_count})"
            ):
                q = item["question"]
                q_type = item["question_type"]
                gt_contexts = item["ground_truth_context"]
                gt_answer = item["ground_truth_answer"]

                retrieved_chunks, gen_answer = self.rag_client.query(q)
                retrieval_metrics = self.semantic_evaluator.compute_metrics(
                    gt_contexts, retrieved_chunks, top_k
                )

                ctx_str = (
                    "\n".join(retrieved_chunks[:top_k]) if retrieved_chunks else "Empty Context"
                )
                llm_metrics = self.llm_evaluator.evaluate(q, q_type, ctx_str, gen_answer, gt_answer)

                abstention_score = (
                    10
                    if (q_type == "unanswerable" and llm_metrics.get("correctness_score", 0) >= 8)
                    else (0 if q_type == "unanswerable" else None)
                )

                error_category = self._categorize_error(
                    q_type,
                    retrieval_metrics["hit_at_k"],
                    llm_metrics.get("faithfulness_score", 0),
                    llm_metrics.get("correctness_score", 0),
                )

                result_row = {
                    "question": q,
                    "question_type": q_type,
                    "hit_at_k": retrieval_metrics["hit_at_k"],
                    "mrr": retrieval_metrics["mrr"],
                    "recall_at_k": retrieval_metrics["recall_at_k"],
                    "faithfulness": llm_metrics.get("faithfulness_score", 0),
                    "answer_correctness": llm_metrics.get("correctness_score", 0),
                    "abstention_accuracy": abstention_score,
                    "error_category": error_category,
                    "generated_answer": gen_answer,
                }

                results.append(result_row)
                checkpoint_file.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                checkpoint_file.flush()

        self._generate_report(results)

    def _generate_report(self, results: List[Dict]):
        df = pd.DataFrame(results)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")

        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION REPORT (Thang điểm 10)")
        logger.info("=" * 50)

        logger.info("\n[OVERALL METRICS]")
        cols = ["hit_at_k", "mrr", "recall_at_k", "faithfulness", "answer_correctness"]
        for col in cols:
            logger.info(f"- {col.upper()}: {df[col].mean():.2f}")

        unanswerable_df = df[df["question_type"] == "unanswerable"]
        if not unanswerable_df.empty:
            logger.info(
                f"- ABSTENTION_ACCURACY: {unanswerable_df['abstention_accuracy'].mean():.2f}"
            )

        logger.info("\n[METRICS BY QUESTION TYPE]")
        breakdown = df.groupby("question_type")[cols].mean().round(2)
        logger.info("\n" + breakdown.to_string())

        logger.info("\n[ERROR ROOT CAUSE DISTRIBUTION]")
        error_counts = df["error_category"].value_counts()
        for err_type, count in error_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"- {err_type}: {count} mẫu ({percentage:.1f}%)")

        logger.info(f"\nSaved at: {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rag_evaluation_dataset.jsonl",
        help="Đường dẫn file dataset JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation_production_report.csv",
        help="Đường dẫn lưu file CSV kết quả",
    )
    parser.add_argument(
        "--api_url", type=str, default="http://127.0.0.1:8000/ask", help="URL của Backend RAG API"
    )
    parser.add_argument(
        "--session_id", type=str, required=True, help="Session ID đã được nạp dữ liệu trên Qdrant"
    )
    parser.add_argument(
        "--top_k", type=int, default=8, help="Số lượng chunk tối đa lấy về để đánh giá"
    )

    args = parser.parse_args()

    pipeline = EvaluationPipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        api_url=args.api_url,
        session_id=args.session_id,
    )
    pipeline.run(top_k=args.top_k)
