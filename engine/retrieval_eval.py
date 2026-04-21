import asyncio
from typing import List, Dict, Any


class RetrievalEvaluator:
    def __init__(self):
        pass

    # ======================
    # CORE METRICS
    # ======================

    def calculate_hit_rate(self, expected_ids, retrieved_ids, top_k=3):
        if not expected_ids or not retrieved_ids:
            return 0.0

        top = retrieved_ids[:top_k]
        return 1.0 if any(x in top for x in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids, retrieved_ids):
        if not expected_ids or not retrieved_ids:
            return 0.0

        for i, doc in enumerate(retrieved_ids):
            if doc in expected_ids:
                return 1.0 / (i + 1)

        return 0.0

    # ======================
    # PER CASE ANALYSIS
    # ======================

    def evaluate_single(self, case: Dict[str, Any]) -> Dict[str, Any]:

        expected = case.get("expected_retrieval_ids", [])
        retrieved = case.get("retrieved_ids", [])

        hit1 = self.calculate_hit_rate(expected, retrieved, top_k=1)
        hit3 = self.calculate_hit_rate(expected, retrieved, top_k=3)
        hit5 = self.calculate_hit_rate(expected, retrieved, top_k=5)

        mrr = self.calculate_mrr(expected, retrieved)

        # ======================
        # FAILURE CLASSIFICATION
        # ======================

        if hit1 == 1:
            category = "perfect_retrieval"
        elif hit3 == 1:
            category = "good_retrieval"
        elif hit5 == 1:
            category = "weak_retrieval"
        else:
            category = "failed_retrieval"

        return {
            "question": case.get("question", ""),
            "expected_ids": expected,
            "retrieved_ids": retrieved,

            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "mrr": mrr,

            "category": category
        }

    # ======================
    # TRUE ASYNC BATCH
    # ======================

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict[str, Any]:

        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(None, self.evaluate_single, case)
            for case in dataset
        ]

        results = await asyncio.gather(*tasks)

        n = len(results)

        avg_hit1 = sum(r["hit@1"] for r in results) / n
        avg_hit3 = sum(r["hit@3"] for r in results) / n
        avg_hit5 = sum(r["hit@5"] for r in results) / n
        avg_mrr = sum(r["mrr"] for r in results) / n

        # ======================
        # FAILURE INSIGHT
        # ======================

        distribution = {
            "perfect": len([r for r in results if r["category"] == "perfect_retrieval"]),
            "good": len([r for r in results if r["category"] == "good_retrieval"]),
            "weak": len([r for r in results if r["category"] == "weak_retrieval"]),
            "failed": len([r for r in results if r["category"] == "failed_retrieval"]),
        }

        return {
            "avg_hit@1": round(avg_hit1, 3),
            "avg_hit@3": round(avg_hit3, 3),
            "avg_hit@5": round(avg_hit5, 3),
            "avg_mrr": round(avg_mrr, 3),

            "total_cases": n,
            "failure_distribution": distribution,
            "details": results
        }