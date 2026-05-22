import pytest

from pipeline.runner import PipelineRunner
from evaluation.evaluator import Evaluator
from decision.hallucination_decider import HallucinationDecider


class MockModel:
    class Out:
        def __init__(self):
            self.responses = ["Answer A", "Answer A", "Answer B"]
            self.token_ids = [[1, 2, 3], [1, 2, 3], [1, 2, 4]]
            self.logits = [None, None, None]
            self.log_probs = [[-0.1, -0.2], [-0.1, -0.2], [-0.3, -0.4]]

    def generate(self, prompt, **kwargs):
        return MockModel.Out()


def test_pipeline_runner_basic_flow():
    model = MockModel()

    # uncertainty module that returns whitebox-like metrics
    def wb_builder(output):
        class M:
            def compute(self_inner):
                return {
                    "white_entropy": 1.0,
                    "white_confidence": 0.9,
                    "white_consistency": 0.8,
                }

        return M()

    uncertainty_modules = {"whitebox": wb_builder}

    evaluator = Evaluator()
    decider = HallucinationDecider({"entropy": 5.0, "confidence": 0.1, "consistency": 0.1, "risk": 2.0})

    runner = PipelineRunner(model=model, uncertainty_modules=uncertainty_modules, evaluator=evaluator, decider=decider)

    result = runner.run("Test prompt")

    assert "responses" in result
    assert "uncertainty" in result
    assert "evaluation" in result
    assert "decision" in result
    assert result["decision"] in ("reliable", "hallucination")
