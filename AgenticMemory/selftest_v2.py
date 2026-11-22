"""Self-test of the patched memory layer with a scripted fake LLM (no Ollama, no GPU, about a minute).

    python selftest_v2.py
"""

import os
os.environ["AMEM_VARIANT"] = ""

import amem_variants as V
from memory_layer_robust_v2 import RobustMemoryNote, RobustAgenticMemorySystem


class FakeRetriever:
    def __init__(self, order):
        self.order = order

    def search(self, query, k=5):
        return list(self.order[:k])

    def add_documents(self, docs):
        pass


class FakeLLM:
    """Scripted responses; records the prompts it receives."""

    def __init__(self, strengthen, update, decision="STRENGTHEN_AND_UPDATE"):
        self.strengthen, self.update, self.decision = strengthen, update, decision
        self.prompts = []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        if "memory evolution agent" in prompt:
            return "DECISION: %s\nREASON: test" % self.decision
        if "provide updated connections" in prompt:
            return self.strengthen
        if "update each neighbor" in prompt:
            return self.update
        return ""


class FakeController:
    def __init__(self, llm):
        self.llm = llm


def set_flags(*flags):
    V.FLAGS.clear()
    V.FLAGS.update(flags)
    V.STATS.clear()


def make_system(llm, n=6):
    ms = object.__new__(RobustAgenticMemorySystem)
    ms.memories, ms.refresh_index, ms.evo_cnt, ms.evo_threshold = {}, False, 0, 100
    ms.retriever = FakeRetriever(list(range(n)))
    ms.llm_controller = FakeController(llm)
    for i in range(n):
        nt = RobustMemoryNote(content="old note %d" % i, keywords=["k%d" % i], context="orig context %d" % i,
                              tags=["orig%d" % i], timestamp="t%d" % i)
        ms.memories[nt.id] = nt
    return ms


def new_note():
    return RobustMemoryNote(content="new note", keywords=["kn"], context="new ctx", tags=["newtag"], timestamp="tn")


ECHO_STRENGTHEN = "CONNECTIONS: 0, 2, 3\nTAGS: tag1, tag2, tag3"
ECHO_UPDATE = (
    "NEIGHBOR 0:\nCONTEXT: updated context sentence\nTAGS: tag1, tag2, tag3\n\n"
    "NEIGHBOR 1:\nCONTEXT: Real new context.\nTAGS: travel, family\n\n"
    "NEIGHBOR 2:\nCONTEXT: KEEP\nTAGS: KEEP\n\n"
    "NEIGHBOR 3:\nCONTEXT: updated context sentence\nTAGS: updated tag1, updated tag2\n\n"
    "NEIGHBOR 4:\nCONTEXT: Another real context.\nTAGS: unchanged tags\n"
)


def run_case(flags, strengthen=ECHO_STRENGTHEN, update=ECHO_UPDATE, decision="STRENGTHEN_AND_UPDATE"):
    set_flags(*flags)
    llm = FakeLLM(strengthen, update, decision)
    ms = make_system(llm)
    notes = list(ms.memories.values())
    nt = new_note()
    label, nt = ms.process_memory(nt)
    return ms, notes, nt, label, llm


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def main():
    print("amem_variants unit tests")
    V._selftest()

    print("\n1. baseline (no switches): behaves like the original and counts the echoes")
    ms, notes, nt, label, llm = run_case([])
    check(label is True, "evolution reported")
    check(nt.tags == ["tag1", "tag2", "tag3"], "template tags are written (original behaviour)")
    check(notes[0].context == "updated context sentence", "template context is written to a neighbour")
    check(len(nt.links) == 3, "the example connections '0, 2, 3' create 3 links")
    check(V.STATS["update_context_echo"] == 2 and V.STATS["update_tags_echo"] == 2,
          "update echoes counted (2 contexts, 2 tag lists)")
    check(V.STATS["strengthen_conn_example"] == 1 and V.STATS["strengthen_tags_echo"] == 1,
          "strengthen echoes counted")

    print("\n2. validate: template text is rejected, real updates are kept")
    ms, notes, nt, label, llm = run_case(["validate"])
    check(nt.tags == ["newtag"], "strengthen tags echo rejected, previous tags kept")
    check(notes[0].context == "orig context 0" and notes[0].tags == ["orig0"], "neighbour 0 keeps its values")
    check(notes[1].context == "Real new context." and notes[1].tags == ["travel", "family"],
          "neighbour 1 receives the genuine update")
    check(notes[2].context == "orig context 2" and notes[2].tags == ["orig2"], "KEEP means no change")
    check(notes[3].context == "orig context 3" and notes[3].tags == ["orig3"], "neighbour 3 echo rejected")
    check(notes[4].context == "Another real context." and notes[4].tags == ["orig4"],
          "'unchanged tags' keeps the tags, genuine context is applied")
    check(nt.links == [], "the literal example connections are dropped")

    check(V.STATS["decision_STRENGTHEN_AND_UPDATE"] == 1, "evolution decisions are counted")
    set_flags("bt0", "noevo"); V.TAG = "n1"
    check(V.cache_suffix() == "_bt0-noevo_n1", "AMEM_TAG is appended to the cache folder suffix")
    V.TAG = ""

    print("\n3. validate + history: overwritten values are recorded")
    ms, notes, nt, label, llm = run_case(["validate", "history"])
    check(len(notes[1].evolution_history) == 2, "neighbour 1 history has the context and the tags change")
    ctx_entry = [h for h in notes[1].evolution_history if h["field"] == "context"][0]
    check(ctx_entry["old"] == "orig context 1" and ctx_entry["new"] == "Real new context." and
          ctx_entry["source_note"] == nt.id, "history keeps the old value, the new value and the source note")
    check(notes[0].evolution_history == [], "no history for a rejected update")

    print("\n4. noupdate / noevo")
    ms, notes, nt, label, llm = run_case(["noupdate"])
    check(all(n.context.startswith("orig") for n in notes), "noupdate: no neighbour is changed")
    check(not any("update each neighbor" in p for p in llm.prompts), "noupdate: the update call is skipped")
    ms, notes, nt, label, llm = run_case(["noevo"])
    check(label is False and llm.prompts == [], "noevo: no LLM call at all")

    print("\n5. harden: the prompts contain no literal example values")
    ms, notes, nt, label, llm = run_case(["harden", "validate"],
                                         strengthen="CONNECTIONS: 1, 4\nTAGS: family, plans",
                                         update=ECHO_UPDATE)
    joined = "\n".join(llm.prompts)
    check("tag1" not in joined and "updated context sentence" not in joined, "no 'tag1' / 'updated context sentence'")
    check("0, 2, 3" not in joined, "no literal '0, 2, 3'")
    check(len(nt.links) == 2, "genuine connections are kept under harden")

    print("\n6. temperature switches")
    set_flags("bt0", "at0")
    V.PHASE = "build"; check(V.effective_temperature(0.7) == 0.0, "build temperature forced to 0")
    V.PHASE = "answer"; check(V.effective_temperature(0.7) == 0.0, "answer temperature forced to 0")
    set_flags(); check(V.effective_temperature(0.7) == 0.7, "unchanged without switches")

    print("\n7. retrieval variants")
    set_flags()
    ms = make_system(FakeLLM("", ""), n=4)
    notes = list(ms.memories.values())
    notes[0].links = [1, 1, 2]       # duplicate link
    notes[3].links = [1]
    ms.retriever = FakeRetriever([0, 1, 3])
    base = ms.find_related_memories_raw("q", k=3)
    check(base.count("memory content:") == 3 + 3 + 0 + 1, "original path: duplicates are listed as before")
    set_flags("dedupe")
    out = ms.find_related_memories_raw("q", k=3)
    check(out.count("memory content: old note 1") == 1, "dedupe: each note appears once")
    set_flags("dedupe", "rev")
    ms.retriever = FakeRetriever([1])
    out = ms.find_related_memories_raw("q", k=1)
    check("old note 0" in out and "old note 3" in out, "rev: notes linking to the retrieved note are added")
    set_flags("ctxmin")
    out = ms.find_related_memories_raw("q", k=1)
    check("memory context:" not in out and "memory content:" in out, "ctxmin: only timestamp and content are shown")

    print("\n8. read-time text repairs (tagfix, datefix) with re-embedding")
    import memory_layer
    captured = {}

    class FakeRet:
        def __init__(self, name):
            captured["name"] = name

        def add_documents(self, docs):
            captured["docs"] = list(docs)

    memory_layer.SimpleEmbeddingRetriever = FakeRet
    set_flags("reembed", "tagfix", "datefix")
    ms = make_system(FakeLLM("", ""), n=2)
    notes = list(ms.memories.values())
    notes[0].content, notes[0].timestamp = "I went yesterday", "1:56 pm on 8 May, 2023"
    notes[0].tags = ["['Marketing", "Social']"]
    ms.retriever = object()                       # no .model attribute: the default encoder name is used
    agent = type("Agent", (), {})()
    agent.memory_system = ms
    V.apply_read_variants(agent)
    check(notes[0].content == "I went yesterday (7 May 2023)", "datefix resolves 'yesterday' from the session date")
    check(notes[0].tags == ["Marketing", "Social"], "tagfix removes brackets and quote marks")
    check(notes[1].content == "old note 1" and notes[1].tags == ["orig1"], "other notes are left alone")
    check(len(captured["docs"]) == 2 and "7 May 2023" in captured["docs"][0], "the index is rebuilt from the repaired text")
    check(V.STATS["datefix_notes"] == 1 and V.STATS["tagfix_notes"] == 1, "repairs are counted")

    print("\n9. hybrid retrieval")
    import numpy as np
    import rank_bm25

    class FakeBM25:
        def __init__(self, docs):
            self.docs = docs

        def get_scores(self, query):
            return [float(sum(w in d for w in query)) for d in self.docs]

    rank_bm25.BM25Okapi = FakeBM25

    class FakeModel:
        def encode(self, texts):
            return np.array([[1.0, 0.0] for _ in texts])      # the query points at dimension 0

    class FakeDense:
        model = FakeModel()
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.9]])   # dense favours note 0

    docs = ["alpha beta", "gamma delta", "epsilon zeta"]
    only_bm25 = V.HybridSearch(FakeDense(), docs, weight=1.0).search("gamma", 2)
    only_dense = V.HybridSearch(FakeDense(), docs, weight=0.0).search("gamma", 2)
    check(int(only_bm25[0]) == 1, "weight 1.0: the keyword match (BM25) ranks first")
    check(int(only_dense[0]) == 0, "weight 0.0: the dense score ranks first")
    mixed = V.HybridSearch(FakeDense(), docs, weight=0.5).search("gamma", 3)
    check(sorted(int(i) for i in mixed) == [0, 1, 2] and len(mixed) == 3, "mixed weights return k distinct notes")

    set_flags("hybrid")
    ms = make_system(FakeLLM("", ""), n=3)
    ms.retriever = object()
    agent = type("Agent", (), {})()
    agent.memory_system = ms
    V.apply_read_variants(agent)
    check(isinstance(ms.retriever, V.HybridSearch) and len(ms.retriever.corpus) == 3,
          "the hybrid switch replaces the retriever of a loaded agent")

    set_flags()
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
