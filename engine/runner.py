import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    # =========================
    # SINGLE TEST EXECUTION
    # =========================

    async def run_single_test(self, test_case: Dict) -> Dict:

        start = time.perf_counter()

        # 1. Agent inference
        response = await self.agent.query(test_case["question"])

        latency = time.perf_counter() - start

        # 2. Retrieval / RAG metrics
        ragas_scores = await self.evaluator.score(test_case, response)

        # 3. Multi-judge evaluation
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"]
        )

        # =========================
        # COST ESTIMATION (IMPORTANT)
        # =========================

        tokens = response.get("metadata", {}).get("tokens_used", 0)
        estimated_cost = tokens * 0.00001  # simple proxy

        return {
            "question": test_case["question"],
            "answer": response["answer"],

            "latency": latency,
            "tokens": tokens,
            "cost": estimated_cost,

            "ragas": ragas_scores,
            "judge": judge_result,

            "status": "pass" if judge_result["final_score"] >= 3 else "fail"
        }

    # =========================
    # BATCH EXECUTION (ASYNC)
    # =========================

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> Dict:

        results = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            tasks = [
                self.run_single_test(case)
                for case in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        # =========================
        # AGGREGATED METRICS
        # =========================

        n = len(results)

        avg_latency = sum(r["latency"] for r in results) / n
        avg_score = sum(r["judge"]["final_score"] for r in results) / n
        pass_rate = sum(1 for r in results if r["status"] == "pass") / n
        total_cost = sum(r["cost"] for r in results)

        failure_cases = [r for r in results if r["status"] == "fail"]

        return {
            "summary": {
                "total_cases": n,
                "avg_latency": round(avg_latency, 3),
                "avg_score": round(avg_score, 3),
                "pass_rate": round(pass_rate, 3),
                "total_cost": round(total_cost, 6)
            },
            "failures": failure_cases,
            "details": results
        }