
"""R4 orchestrator: execute plan waves and manage worker dependencies."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from src.router.schemas import Plan
from src.workers.worker import RetrievalWorker


class WorkerOrchestrator:
    def __init__(
        self,
        worker: RetrievalWorker,
        max_parallel_workers: int = 4,
    ):
        self.worker = worker
        self.max_parallel_workers = max(1, max_parallel_workers)

    def run(self, plan: Plan) -> Dict[str, Any]:
        """Execute independent sub-questions concurrently, wave by wave."""
        answers: Dict[str, str] = {}
        results: Dict[str, Dict[str, Any]] = {}
        trace = []

        for wave_number, wave in enumerate(plan.waves(), start=1):
            def execute(sub_question):
                resolved_question = sub_question.resolve(answers)
                return self.worker.run(sub_question, resolved_question)

            with ThreadPoolExecutor(
                max_workers=self.max_parallel_workers
            ) as executor:
                futures = {
                    executor.submit(execute, sub_question): sub_question
                    for sub_question in wave
                }

                wave_results = []
                for future in as_completed(futures):
                    sub_question = futures[future]
                    try:
                        result = future.result()
                        wave_results.append(result)
                    except Exception as exc:
                        wave_results.append({
                            "sub_question_id": sub_question.id,
                            "question": sub_question.text,
                            "error": str(exc),
                            "chunks": [],
                            "context": "",
                            "estimated_tokens": 0,
                            "chunk_count": 0,
                        })

            # Update answers only after the whole wave completes.
            # This ensures same-wave workers cannot depend on each other.
            for result in wave_results:
                sid = result["sub_question_id"]
                results[sid] = result
                answers[sid] = result.get("context", "")

            trace.append({
                "wave": wave_number,
                "sub_question_ids": [s.id for s in wave],
                "completed": len(wave_results),
                "failed": sum("error" in r for r in wave_results),
            })

        return {
            "question": plan.question,
            "results": results,
            "answers": answers,
            "trace": trace,
            "failed_workers": [
                sid for sid, result in results.items() if "error" in result
            ],
        }
