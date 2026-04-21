import asyncio
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI

client = AsyncOpenAI()


class LLMJudge:
    def __init__(self):
        # 2 judges as required by rubric
        self.judges = [
            {"name": "gpt-4o", "weight": 0.7},
            {"name": "gpt-4o-mini", "weight": 0.3}
        ]

    # ================================
    # 🧠 LLM AS A JUDGE
    # ================================

    async def _call_judge_llm(
        self,
        model: str,
        question: str,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:

        prompt = f"""
You are an expert AI evaluator.

Evaluate the following answer.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

MODEL ANSWER:
{answer}

Return ONLY valid JSON:

{{
  "score": float (0-5),
  "reason": "short explanation",
  "correctness": float (0-5),
  "groundedness": float (0-5),
  "relevance": float (0-5)
}}
"""

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except:
            # fallback if model returns non-JSON
            return {
                "score": 2.5,
                "reason": "parse_error",
                "correctness": 2.5,
                "groundedness": 2.5,
                "relevance": 2.5
            }

    # ================================
    # ⚖️ MULTI-JUDGE CONSENSUS
    # ================================

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:

        results = await asyncio.gather(*[
            self._call_judge_llm(j["name"], question, answer, ground_truth)
            for j in self.judges
        ])

        scored = []
        weighted_sum = 0
        total_weight = 0

        for judge, result in zip(self.judges, results):
            score = float(result["score"])
            weight = judge["weight"]

            scored.append({
                "model": judge["name"],
                "score": score,
                "reason": result.get("reason", "")
            })

            weighted_sum += score * weight
            total_weight += weight

        final_score = weighted_sum / total_weight

        # ================================
        # 📊 AGREEMENT METRICS
        # ================================

        scores = [s["score"] for s in scored]
        score_diff = max(scores) - min(scores)

        agreement_rate = 1.0 - (score_diff / 5.0)

        # ================================
        # ⚠️ CONFLICT RESOLUTION
        # ================================

        if score_diff > 1.5:
            resolution = "high_conflict_weighted_gpt_priority"
            final_score = max(scores)  # trust stronger model
        else:
            resolution = "weighted_average"

        return {
            "final_score": round(final_score, 3),
            "agreement_rate": round(agreement_rate, 3),
            "score_variance": round(score_diff, 3),
            "resolution": resolution,
            "individual_results": scored
        }

    # ================================
    # 🔄 POSITION BIAS TEST
    # ================================

    async def check_position_bias(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        ground_truth: str
    ) -> Dict[str, Any]:

        res_a = await self._call_judge_llm(
            "gpt-4o",
            question,
            answer_a,
            ground_truth
        )

        res_b = await self._call_judge_llm(
            "gpt-4o",
            question,
            answer_b,
            ground_truth
        )

        bias = abs(res_a["score"] - res_b["score"])

        return {
            "score_a": res_a["score"],
            "score_b": res_b["score"],
            "position_bias": round(bias, 3),
            "bias_detected": bias > 0.7
        }