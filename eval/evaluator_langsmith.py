"""Из корня репозитория: python -m eval.evaluator_langsmith data/test.jsonl

Датасет hse-rag-<имя файла> создается в LangSmith из jsonl, если его еще нет.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from langsmith import Client

from eval.eval_functions import score_context
from eval.evaluator_excel import judge, wait_for_window
from eval.rag_example import RAG


def metrics(outputs, reference_outputs):
    context = score_context(reference_outputs["gold_context"], outputs["context"])
    answer = judge(reference_outputs["gold_answer"], outputs["answer"])
    return [
        {"key": "context_recall", "score": context["context_recall"]},
        {"key": "context_precision", "score": context["context_precision"]},
        {"key": "answer_correctness", "score": answer["answer_correctness"], "comment": answer["reason"]},
        {"key": "time", "score": outputs["time"]},
    ]


def ensure_dataset(client, path):
    name = "hse-rag-" + path.stem
    if client.has_dataset(dataset_name=name):
        return name
    dataset = client.create_dataset(name)
    items = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": item["question"]} for item in items],
        outputs=[{"gold_answer": item["gold_answer"], "gold_context": item["gold_context"]} for item in items],
        metadata=[{"question_id": item["question_id"]} for item in items],
    )
    return name


def evaluate(dataset):
    client = Client()
    if dataset.endswith(".jsonl"):
        dataset = ensure_dataset(client, Path(dataset))
    system = RAG()

    def answer_question(inputs):
        wait_for_window()
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    results = client.evaluate(
        answer_question, data=dataset, evaluators=[metrics], experiment_prefix="rag",
    )
    share = client.share_dataset(dataset_name=dataset)
    print("эксперимент:", results.experiment_name)
    print("публичная ссылка:", share)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="имя датасета в LangSmith или путь к jsonl")
    evaluate(parser.parse_args().dataset)
