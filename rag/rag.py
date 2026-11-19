"""Agentic RAG по отчётам банков.

1. По названию банка в вопросе выбираем его отчёты, по годам и «6 месяцев» поднимаем нужные.
2. Гибридный поиск по страницам: BM25 + эмбеддинги, ранги складываем (RRF).
3. LLM видит найденные страницы и сама решает: ответить (answer) или поискать ещё (search).
   Контекст ответа это страницы, которые модель указала как источники.
"""

import json
import math
import os
import re
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache

import numpy as np
import requests
from langsmith import traceable  # трассы пишутся, только если задан LANGSMITH_TRACING=true

from rag.index import VECTORS, doc_title, embed
from rag.parse import DATA, OUT as PAGES

LLM_URL = "https://claude.gigaparsers.ru/v1/messages"
MODEL = "deepseek-flash"
PAGE_CHARS = 4000
MAX_SEARCHES = 2

BANKS = {
    "sber": r"сбер",
    "vtb": r"втб",
    "tbank": r"т-?банк|тинькофф",
    "domrf": r"дом\.?\s?рф",
    "mkb": r"мкб|московск\w* кредитн",
    "rshb": r"россельхоз|сельскохозяйственн|рсхб",
    "sovkom": r"совком",
    "alfa": r"альфа",
    "gpb": r"газпромбанк|нпф",
}

SYSTEM = """Ты аналитик финансовой отчётности банков по МСФО. Отвечай только по страницам отчётов, которые тебе показали.

Страницы подписаны так: [doc_id стр. N]. В таблицах ячейки разделены « | », первые колонки обычно отчётный период, следующие сравнительный.

Как отвечать:
- Коротко: только то, о чём спрашивают, без справочных цифр за другие периоды и лишних подробностей. Если просят перечислить, перечисли все пункты из отчёта и ничего не добавляй. Числа пиши как в отчёте, с единицами измерения (млн руб., млрд руб., тыс. белорусских рублей) и датой или периодом. Не округляй.
- Если спрашивают об изменении, сравнении или доле, приведи оба исходных значения и посчитай разницу или долю.
- В sources укажи только страницы, откуда взяты числа и факты ответа, обычно одну или две. Не перечисляй лишние.
- Проверь, что период, организация и показатель совпадают с вопросом. Цифры за прошлый год есть в сравнительной колонке следующего отчёта, но лучше брать отчёт за тот период, о котором вопрос.

Если нужной таблицы или строки среди страниц нет, вызови search с другим запросом (названием строки или раздела, как в отчёте) и нужными doc_id. Когда данных хватает, вызови answer."""

TOOLS = [
    {
        "name": "search",
        "description": "Найти ещё страницы отчётов. query: формулировка как в отчёте; doc_ids: в каких отчётах искать.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query", "doc_ids"],
        },
    },
    {
        "name": "answer",
        "description": "Финальный ответ и страницы-источники.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer"}},
                        "required": ["doc_id", "page"],
                    },
                },
            },
            "required": ["answer", "sources"],
        },
    },
]

# Время всех вызовов LLM в процессе: eval по нему ждёт свободного окна до старта вопроса
CALLS = deque(maxlen=100)


@traceable(run_type="llm", name="deepseek-flash")
def llm(messages, tool_choice):
    body = {
        "model": MODEL,
        "max_tokens": 1500,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": SYSTEM,
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "messages": messages,
    }
    for attempt in range(6):
        CALLS.append(time.time())
        try:
            response = requests.post(
                LLM_URL,
                headers={"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"},
                json=body,
                timeout=60,
            )
        except requests.RequestException:
            if attempt == 5:
                raise
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 5 * (attempt + 1))))
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()


def tokens(text):
    return [w[:6] for w in re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))]


class RAG:
    def __init__(self):
        self.docs = {d["doc_id"]: d for d in map(json.loads, (DATA / "documents.jsonl").open(encoding="utf-8"))}
        self.pages = [json.loads(line) for line in PAGES.open(encoding="utf-8")]
        self.where = {(p["doc_id"], p["page"]): i for i, p in enumerate(self.pages)}
        # без файла эмбеддингов работает только BM25
        self.vectors = np.load(VECTORS).astype(np.float32) if VECTORS.exists() else None
        self.postings = defaultdict(list)
        lengths = []
        for i, page in enumerate(self.pages):
            counts = Counter(tokens(page["text"]))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avg_length = self.lengths.mean()

    def bm25(self, query, k1=1.5, b=0.75):
        scores = np.zeros(len(self.pages), dtype=np.float32)
        n = len(self.pages)
        for term in set(tokens(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lengths[i] / self.avg_length))
        return scores

    @lru_cache(maxsize=512)
    def query_vector(self, query):
        return embed(query)

    @traceable(run_type="retriever")
    def search(self, query, doc_ids, k, prior=None):
        candidates = [i for i, p in enumerate(self.pages) if p["doc_id"] in doc_ids]
        if not candidates:
            return []
        signals = [self.bm25(query)[candidates]]
        if self.vectors is not None:
            signals.append(self.vectors[candidates] @ self.query_vector(query))
        score = np.zeros(len(candidates))
        for values in signals:
            ranks = np.empty(len(candidates))
            ranks[np.argsort(-values)] = np.arange(len(candidates))
            score += 1 / (60 + ranks)
        if prior:
            score += np.array([prior.get(self.pages[i]["doc_id"], 0) for i in candidates])
        return [candidates[j] for j in np.argsort(-score)[:k]]

    def route(self, question):
        """Отчёты банков из вопроса и бонус отчётам нужного года и периода."""
        low = question.lower()
        banks = [b for b, pattern in BANKS.items() if re.search(pattern, low)] or list(BANKS)
        years = set(re.findall(r"20(2[2-5])", question))
        half = bool(re.search(r"шест\w* месяц|6 месяц|полугод|30 июня", low))
        prior = {}
        for doc_id in self.docs:
            bank, year, kind = doc_id.split("_")
            if bank in banks and year[2:] in years:
                prior[doc_id] = 0.01 + (0.005 if (kind == "6m") == half else 0)
        return banks, prior

    def show(self, indices):
        return "\n\n".join(
            "[%s стр. %d] %s\n%s" % (self.pages[i]["doc_id"], self.pages[i]["page"],
                                    doc_title(self.docs[self.pages[i]["doc_id"]]),
                                    self.pages[i]["text"][:PAGE_CHARS])
            for i in indices
        )

    @traceable(name="rag")
    def rag(self, question):
        banks, prior = self.route(question)
        per_bank = 6 if len(banks) == 1 else 4
        found = []
        for bank in banks if len(banks) <= 3 else [None]:
            ids = [d for d in self.docs if bank is None or d.startswith(bank + "_")]
            found += self.search(question, ids, per_bank if bank else 6, prior)
        doc_list = ", ".join(d for d in self.docs if d.split("_")[0] in banks)
        messages = [{
            "role": "user",
            "content": "Вопрос: %s\n\nОтчёты для поиска: %s\n\nНайденные страницы:\n\n%s"
                       % (question, doc_list, self.show(found)),
        }]
        seen = set(found)
        for step in range(MAX_SEARCHES + 1):
            last = step == MAX_SEARCHES
            reply = llm(messages, {"type": "tool", "name": "answer"} if last else {"type": "any"})
            calls = [b for b in reply["content"] if b["type"] == "tool_use"]
            final = next((c for c in calls if c["name"] == "answer"), None)
            if final or not calls:
                break
            messages.append({"role": "assistant", "content": reply["content"]})
            results = []
            for call in calls:
                ids = [d for d in call["input"].get("doc_ids", []) if d in self.docs] or list(self.docs)
                new = [i for i in self.search(call["input"].get("query", question), ids, 4) if i not in seen]
                seen.update(new)
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": self.show(new) or "Новых страниц не найдено."})
            messages.append({"role": "user", "content": results})
        if not final:  # модель ответила текстом без answer
            text = "".join(b.get("text", "") for b in reply["content"] if b["type"] == "text")
            return {"answer": text.strip(), "context": self.context([], found)}
        args = final["input"]
        return {"answer": str(args.get("answer") or ""), "context": self.context(args.get("sources", []), found)}

    def context(self, sources, found):
        pages = defaultdict(list)
        for s in sources if isinstance(sources, list) else []:
            try:
                key = (s["doc_id"], int(s["page"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in self.where and key[1] not in pages[key[0]]:
                pages[key[0]].append(key[1])
        if not pages and found:
            top = self.pages[found[0]]
            pages[top["doc_id"]].append(top["page"])
        return [{"doc_id": d, "pages": sorted(p)} for d, p in pages.items()]
