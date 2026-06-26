import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper

from src.retrieval.retriever import retrieve
from src.generation.generator import answer

EVAL_SET_PATH = "data/eval_set.json"


def build_samples() -> list[SingleTurnSample]:
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    samples = []
    for i, item in enumerate(eval_set, 1):
        print(f"Running question {i}/{len(eval_set)}: {item['question'][:60]}...")

        result = answer(item["question"], k=5)
        docs = retrieve(item["question"], k=5)

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=[doc.page_content for doc in docs],
            reference=item["reference"],
        ))

    return samples


def run_evaluation():
    load_dotenv()

    print("Building samples — running all eval_set.json 12 questions through the pipeline...\n")
    samples = build_samples()

    print("\nScoring with Ragas...\n")
    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ))

    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(), Faithfulness(), ResponseRelevancy()],
        llm=judge_llm,
    )

    print("\n=== BASELINE SCORES ===")
    print(result)
    return result


if __name__ == "__main__":
    run_evaluation()
