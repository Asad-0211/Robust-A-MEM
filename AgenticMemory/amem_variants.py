"""Variant switches, echo validator, hardened prompts and read-time transforms for the A-MEM reliability study.

The behaviour is selected with the environment variable AMEM_VARIANT, a comma-separated list of switches.
With the variable empty, every hook is a no-op and the code behaves exactly like the original.

Build-time switches (they change how memories are built; each combination gets its own cache folder):
    noevo       no evolution at all (no decision call, no links, no neighbour updates)
    noupdate    keep link generation and tag strengthening, skip the neighbour-update call
    harden      prompts without literal example values (wording only; see HARD_* below)
    validate    reject template text and the literal example connections, keep the previous values
    history     record every overwritten value in note.evolution_history
    bt0         temperature 0 for every construction call

Read-time switches (applied to cached memories when answering; they reuse the baseline cache):
    reembed     rebuild the retriever from the cached notes with note_document
    blank       blank template annotations (they are then shown and embedded blank); implies reembed
    embcontent  embed the note content only; implies reembed
    ctxmin      show only timestamp and content in the answering context
    dedupe      list each note once in the expanded context
    rev         also expand the context through reverse links
    qplus       retrieval query = question + generated keywords
    tagfix      clean stray brackets and quote marks in tags; implies reembed
    datefix     append the resolved date after relative time words ("yesterday (7 May 2023)") using each note's
                session date; implies reembed
    hybrid      retrieve with min-max-normalised BM25 + dense scores (weight of BM25: AMEM_BM25_WEIGHT, default
                0.5); implies reembed
    at0         temperature 0 for every answering call

AMEM_TAG (optional, separate variable): appended to the cache folder name, so that repeated builds with identical
switches (noise-floor runs) do not share a cache.
"""

import os
import re
from collections import Counter

BUILD_FLAGS = {"noevo", "noupdate", "harden", "validate", "history", "bt0"}
READ_FLAGS = {"reembed", "blank", "embcontent", "ctxmin", "dedupe", "rev", "qplus", "at0", "tagfix", "datefix", "hybrid"}

FLAGS = {t.strip() for t in os.environ.get("AMEM_VARIANT", "").split(",") if t.strip()}
_unknown = FLAGS - BUILD_FLAGS - READ_FLAGS
if _unknown:
    raise SystemExit("Unknown AMEM_VARIANT switch(es): %s" % sorted(_unknown))

BM25_WEIGHT = float(os.environ.get("AMEM_BM25_WEIGHT", "0.5"))   # weight of BM25 in the hybrid switch
NBR_CAP = int(os.environ.get("AMEM_NBR_CAP", "5"))   # linked notes added per retrieved note (dedupe/rev/ctxmin path)
STATS = Counter()                                     # counters written to the results file
PHASE = "answer"                                      # the driver sets "build" while it builds memories


def has(flag):
    return flag in FLAGS


TAG = os.environ.get("AMEM_TAG", "").strip()          # separates repeated builds that use identical switches


def cache_suffix():
    """Cache-folder suffix: switches that change how memories are built, plus an optional AMEM_TAG."""
    build = sorted(FLAGS & BUILD_FLAGS)
    suffix = ("_" + "-".join(build)) if build else ""
    return suffix + (("_" + TAG) if TAG else "")


def effective_temperature(temperature):
    if PHASE == "build" and "bt0" in FLAGS:
        return 0.0
    if PHASE == "answer" and "at0" in FLAGS:
        return 0.0
    return temperature


# ---------------------------------------------------------------------------
# Template-text detection
# ---------------------------------------------------------------------------

_TEMPLATE_PATTERNS = [
    r"^\W*(updated[\s_]*)?tags?[\s_]*\d\W*$",           # tag1, updated tag2, updated_tag3, [tag1]
    r"^\W*(updated[\s_]*)?keywords?[\s_]*\d\W*$",       # keyword1
    r"updated context sentence",
    r"a single sentence summarizing",
    r"^\W*<[^>]*>\W*$",                                 # an unfilled <...> placeholder
    r"separated by commas",
    r"comma[- ]separated",
    r"^\W*your (keywords|tags|one-sentence)",
]
_KEEP_PATTERNS = [r"^\W*(keep|unchanged( tags| context)?|no changes?( needed)?|same)\W*$"]
_TEMPLATE_RX = [re.compile(p, re.IGNORECASE) for p in _TEMPLATE_PATTERNS]
_KEEP_RX = [re.compile(p, re.IGNORECASE) for p in _KEEP_PATTERNS]


def is_template(text):
    t = str(text).strip()
    return bool(t) and any(rx.search(t) for rx in _TEMPLATE_RX)


def is_keep(text):
    t = str(text).strip()
    return bool(t) and any(rx.search(t) for rx in _KEEP_RX)


# ---------------------------------------------------------------------------
# Vetting hooks. Without "validate" they only count; with it they repair.
# ---------------------------------------------------------------------------

def vet_tags(tags, site):
    """Return the tags to store ([] = keep the previous tags)."""
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
    if not tags:
        return []
    STATS[site + "_total"] += 1
    echo = any(is_template(t) for t in tags)
    keep = all(is_keep(t) for t in tags)
    if echo:
        STATS[site + "_echo"] += 1
    if keep:
        STATS[site + "_keep"] += 1
    if "validate" not in FLAGS:
        return tags
    if keep:
        return []
    good = [t for t in tags if not is_template(t) and not is_keep(t)]
    if len(good) < len(tags):
        STATS[site + "_blocked"] += 1
    return good


def vet_context(text, site):
    """Return the context to store ('' = keep the previous context)."""
    text = str(text or "").strip()
    if not text:
        return ""
    STATS[site + "_total"] += 1
    echo, keep = is_template(text), is_keep(text)
    if echo:
        STATS[site + "_echo"] += 1
    if keep:
        STATS[site + "_keep"] += 1
    if "validate" not in FLAGS:
        return text
    if echo or keep:
        if echo:
            STATS[site + "_blocked"] += 1
        return ""
    return text


def vet_connections(conns):
    """Count (and, with validate and the original prompt, drop) the literal example '0, 2, 3'."""
    conns = list(conns or [])
    if not conns:
        return conns
    STATS["strengthen_conn_total"] += 1
    if conns == [0, 2, 3]:
        STATS["strengthen_conn_example"] += 1
        if "validate" in FLAGS and "harden" not in FLAGS:
            STATS["strengthen_conn_blocked"] += 1
            return []
    return conns


def vet_analysis(analysis, content):
    """Check the note-construction output; with validate, repair template fields from the content."""
    from llm_text_parsers import _heuristic_keywords, _heuristic_context
    out = dict(analysis)
    for field in ("keywords", "tags"):
        vals = [str(v).strip() for v in (out.get(field) or []) if str(v).strip()]
        STATS["analyze_%s_total" % field] += 1
        if any(is_template(v) for v in vals):
            STATS["analyze_%s_echo" % field] += 1
            if "validate" in FLAGS:
                good = [v for v in vals if not is_template(v)]
                out[field] = good if good else (
                    _heuristic_keywords(content) if field == "keywords" else _heuristic_keywords(content, 3))
                STATS["analyze_%s_blocked" % field] += 1
    ctx = str(out.get("context") or "").strip()
    STATS["analyze_context_total"] += 1
    if is_template(ctx):
        STATS["analyze_context_echo"] += 1
        if "validate" in FLAGS:
            out["context"] = _heuristic_context(content)
            STATS["analyze_context_blocked"] += 1
    return out


def log_history(note, field, old, new, source_id):
    """Record an overwritten value in note.evolution_history (switch: history)."""
    if "history" not in FLAGS or old == new:
        return
    note.evolution_history.append({"field": field, "old": old, "new": new, "source_note": source_id})
    STATS["history_entries"] += 1


# ---------------------------------------------------------------------------
# Hardened prompts: the same instructions, without literal example values
# ---------------------------------------------------------------------------

HARD_ANALYZE = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this layout, writing your own content after each header (do not copy the descriptions):

KEYWORDS: <your keywords, separated by commas>
CONTEXT: <your one-sentence summary>
TAGS: <your tags, separated by commas>

Content for analysis:
{content}"""

HARD_STRENGTHEN = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbors should the new memory connect to? Use the "memory index" numbers shown above. What tags best describe this memory?

Respond using EXACTLY this layout, writing your own answer after each header (write NONE if no neighbor is related):

CONNECTIONS: <memory index numbers of related neighbors, separated by commas>
TAGS: <your tags, separated by commas>"""

HARD_UPDATE = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), write an improved one-sentence context and a list of tags. If a neighbor needs no change, write KEEP after both CONTEXT and TAGS.

Respond using EXACTLY this layout, one block per neighbor, writing your own content after each header:

NEIGHBOR 0:
CONTEXT: <improved one-sentence context, or KEEP>
TAGS: <tags separated by commas, or KEEP>

Repeat the block for every neighbor, numbering the blocks 0 to {max_neighbor_idx} ({neighbor_count} neighbors in total)."""


def analyze_prompt(original):
    return HARD_ANALYZE if "harden" in FLAGS else original


def strengthen_prompt(original):
    return HARD_STRENGTHEN if "harden" in FLAGS else original


def update_prompt(original):
    return HARD_UPDATE if "harden" in FLAGS else original



# ---------------------------------------------------------------------------
# Read-time text repairs: tagfix and datefix
# ---------------------------------------------------------------------------

import calendar
import datetime as _dt

_TS_RX = re.compile(r"(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.IGNORECASE)
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_REL_RX = re.compile(
    r"\b(?:(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<unit>day|week|month|year)s?\s+ago"
    r"|last\s+(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?P<rel>yesterday|today|tonight|tomorrow|last\s+night|last\s+week|last\s+month|last\s+year"
    r"|next\s+week|next\s+month|next\s+year))\b", re.IGNORECASE)


def parse_session_date(timestamp):
    """'1:56 pm on 8 May, 2023' -> datetime.date, or None."""
    m = _TS_RX.search(str(timestamp))
    if not m or m.group(5).lower() not in _MONTHS:
        return None
    try:
        return _dt.date(int(m.group(6)), _MONTHS[m.group(5).lower()], int(m.group(4)))
    except ValueError:
        return None


def _fmt(d):
    return "%d %s %d" % (d.day, calendar.month_name[d.month], d.year)


def _add_months(d, n):
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return _dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def resolve_relative_dates(text, timestamp):
    """Append the absolute date after relative time expressions, using the note's session date."""
    base = parse_session_date(timestamp)
    if base is None:
        return text

    def repl(m):
        word = m.group(0)
        if m.group("rel"):
            rel = " ".join(m.group("rel").lower().split())
            out = {
                "yesterday": _fmt(base - _dt.timedelta(days=1)),
                "last night": _fmt(base - _dt.timedelta(days=1)),
                "today": _fmt(base),
                "tonight": _fmt(base),
                "tomorrow": _fmt(base + _dt.timedelta(days=1)),
                "last week": "the week before " + _fmt(base),
                "next week": "the week after " + _fmt(base),
                "last month": "%s %d" % (calendar.month_name[_add_months(base, -1).month], _add_months(base, -1).year),
                "next month": "%s %d" % (calendar.month_name[_add_months(base, 1).month], _add_months(base, 1).year),
                "last year": str(base.year - 1),
                "next year": str(base.year + 1),
            }[rel]
        elif m.group("wd"):
            target = _WEEKDAYS.index(m.group("wd").lower())
            back = (base.weekday() - target) % 7 or 7
            out = _fmt(base - _dt.timedelta(days=back))
        else:
            n = int(m.group("n")) if m.group("n").isdigit() else _NUMWORDS[m.group("n").lower()]
            unit = m.group("unit").lower()
            if unit == "day":
                out = _fmt(base - _dt.timedelta(days=n))
            elif unit == "week":
                out = "around " + _fmt(base - _dt.timedelta(days=7 * n))
            elif unit == "month":
                d = _add_months(base, -n)
                out = "%s %d" % (calendar.month_name[d.month], d.year)
            else:
                out = str(base.year - n)
        return "%s (%s)" % (word, out)

    return _REL_RX.sub(repl, text)


def fix_tags(tags):
    """Strip stray brackets and quote marks from each tag, drop empties and duplicates."""
    out, seen = [], set()
    for t in tags or []:
        c = str(t).strip().strip("[]'\" ").strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


class HybridSearch:
    """Dense + BM25 retrieval behind the search(query, k) interface the memory system uses."""

    def __init__(self, dense, docs, weight=0.5):
        from rank_bm25 import BM25Okapi
        self.dense, self.weight, self.corpus = dense, float(weight), list(docs)
        self.bm25 = BM25Okapi([self._tok(d) for d in self.corpus])

    @staticmethod
    def _tok(text):
        return re.findall(r"\w+", str(text).lower())

    def search(self, query, k=5):
        import numpy as np
        if not self.corpus:
            return []
        q = np.asarray(self.dense.model.encode([query])[0], dtype=float)
        emb = np.asarray(self.dense.embeddings, dtype=float)
        sem = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-12)
        bm = np.asarray(self.bm25.get_scores(self._tok(query)), dtype=float)

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-12)

        score = self.weight * norm(bm) + (1.0 - self.weight) * norm(sem)
        k = min(int(k), len(self.corpus))
        return np.argsort(score)[-k:][::-1]

# ---------------------------------------------------------------------------
# Read-time transforms of cached memories
# ---------------------------------------------------------------------------

def apply_read_variants(agent):
    """blank / reembed / embcontent: rebuild the retriever of a loaded agent from its cached notes."""
    if not (FLAGS & {"reembed", "blank", "embcontent", "tagfix", "datefix", "hybrid"}):
        return
    from memory_layer import SimpleEmbeddingRetriever, note_document
    ms = agent.memory_system
    notes = list(ms.memories.values())
    if "blank" in FLAGS:
        for nt in notes:
            ctx = nt.context if isinstance(nt.context, str) else " ".join(nt.context)
            tags = [str(t) for t in (nt.tags or [])]
            if is_template(ctx) or is_keep(ctx):
                nt.context = ""
                STATS["blank_context"] += 1
            if tags and any(is_template(t) or is_keep(t) for t in tags):
                nt.tags = []
                STATS["blank_tags"] += 1
    if "tagfix" in FLAGS:
        for nt in notes:
            fixed = fix_tags(nt.tags)
            if fixed != list(nt.tags or []):
                nt.tags = fixed
                STATS["tagfix_notes"] += 1
    if "datefix" in FLAGS:
        for nt in notes:
            resolved = resolve_relative_dates(nt.content, nt.timestamp)
            if resolved != nt.content:
                nt.content = resolved
                STATS["datefix_notes"] += 1
    try:
        model_name = ms.retriever.model.get_config_dict()["model_name"]
    except (AttributeError, KeyError):
        model_name = "all-MiniLM-L6-v2"
    ms.retriever = SimpleEmbeddingRetriever(model_name)
    docs = [nt.content if "embcontent" in FLAGS else note_document(nt) for nt in notes]
    ms.retriever.add_documents(docs)
    if "hybrid" in FLAGS:
        ms.retriever = HybridSearch(ms.retriever, docs, BM25_WEIGHT)
    STATS["reembedded_notes"] += len(docs)


# ---------------------------------------------------------------------------
# Self-test: python amem_variants.py
# ---------------------------------------------------------------------------

def _selftest():
    for s in ["tag1", "Tag 2", "updated tag1", "updated_tag3", "[tag1]", "keyword2", "updated context sentence",
              "A single sentence summarizing the content.", "<your tags, separated by commas>"]:
        assert is_template(s), s
    for s in ["Business", "Photography", "travel plans", "Gina opened a clothing store.", "tagline", "Tag Team"]:
        assert not is_template(s), s
    for s in ["KEEP", "keep", "unchanged tags", "No change needed", "same"]:
        assert is_keep(s), s
    assert not is_keep("keeping fit")

    FLAGS.clear(); STATS.clear()
    assert vet_tags(["tag1", "tag2"], "t") == ["tag1", "tag2"]          # counting only
    assert STATS["t_echo"] == 1
    FLAGS.add("validate"); STATS.clear()
    assert vet_tags(["tag1", "tag2"], "t") == []
    assert vet_tags(["travel", "tag2"], "t") == ["travel"]
    assert vet_tags(["KEEP"], "t") == []
    assert vet_context("updated context sentence", "c") == ""
    assert vet_context("Gina opened a store.", "c") == "Gina opened a store."
    assert vet_connections([0, 2, 3]) == [] and vet_connections([1, 4]) == [1, 4]
    FLAGS.add("harden")
    assert vet_connections([0, 2, 3]) == [0, 2, 3]                       # no literal example any more
    FLAGS.clear(); FLAGS.update({"bt0", "at0"})
    global PHASE
    PHASE = "build"; assert effective_temperature(0.7) == 0.0
    PHASE = "answer"; assert effective_temperature(0.7) == 0.0
    FLAGS.clear(); assert effective_temperature(0.7) == 0.7
    FLAGS.update({"harden", "validate", "bt0", "at0"}); assert cache_suffix() == "_bt0-harden-validate"
    ts = "1:56 pm on 8 May, 2023"                              # a Monday
    assert parse_session_date(ts) == _dt.date(2023, 5, 8) and parse_session_date("no date") is None
    assert resolve_relative_dates("I went yesterday.", ts) == "I went yesterday (7 May 2023)."
    assert "last Friday (5 May 2023)" in resolve_relative_dates("It was last Friday.", ts)
    assert "two weeks ago (around 24 April 2023)" in resolve_relative_dates("two weeks ago", ts)
    assert "3 days ago (5 May 2023)" in resolve_relative_dates("3 days ago", ts)
    assert "last month (April 2023)" in resolve_relative_dates("last month", ts)
    assert "last year (2022)" in resolve_relative_dates("last year", ts)
    assert "tomorrow (9 May 2023)" in resolve_relative_dates("see you tomorrow", ts)
    assert resolve_relative_dates("Nothing relative here.", ts) == "Nothing relative here."
    assert resolve_relative_dates("yesterday", "garbled") == "yesterday"
    assert fix_tags(["['Marketing", "SocialMedia", "DanceIndustry']", "marketing"]) == ["Marketing", "SocialMedia", "DanceIndustry"]
    assert fix_tags(["Women's health"]) == ["Women's health"]
    FLAGS.clear(); STATS.clear()
    print("amem_variants self-test passed")


if __name__ == "__main__":
    _selftest()
