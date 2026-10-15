
from src.router.schemas import Plan, SubQuestion
from src.workers.orchestrator import WorkerOrchestrator


class FakeWorker:
    def run(self, sub_question, resolved_question=None):
        question = resolved_question or sub_question.text
        return {
            "sub_question_id": sub_question.id,
            "question": question,
            "chunks": [],
            "context": f"Answer evidence for {question}",
            "estimated_tokens": 5,
            "chunk_count": 0,
        }


def test_orchestrator_runs_all_subquestions():
    plan = Plan(
        question="Compare A and B",
        sub_questions=[
            SubQuestion("q1", "What is A?"),
            SubQuestion("q2", "What is B?"),
        ],
    )

    result = WorkerOrchestrator(FakeWorker()).run(plan)

    assert set(result["results"]) == {"q1", "q2"}
    assert len(result["trace"]) == 1
    assert result["failed_workers"] == []


def test_dependent_question_runs_after_first_wave():
    plan = Plan(
        question="Explain A and its limitations",
        sub_questions=[
            SubQuestion("q1", "What is A?"),
            SubQuestion(
                "q2",
                "What are its limitations? {q1}",
                depends_on=["q1"],
            ),
        ],
    )

    result = WorkerOrchestrator(FakeWorker()).run(plan)

    assert len(result["trace"]) == 2
    assert "Answer evidence for What is A?" in result["results"]["q2"]["question"]
