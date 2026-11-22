"""Create memory_layer_robust_v2.py and test_v2.py from the original files.

The originals are never modified. Every patch is anchored on an exact piece of the original text; if an anchor is
missing or ambiguous, nothing is written and the offending patch is reported.

    python make_v2.py
"""

import sys

# ---------------------------------------------------------------------------
# memory_layer_robust.py -> memory_layer_robust_v2.py
# ---------------------------------------------------------------------------

RAW_V2_METHOD = '''    def _raw_v2(self, query: str, k: int) -> str:
        """Retrieval with optional de-duplication, reverse-link expansion and minimal formatting (variants)."""
        if not self.memories:
            return ""
        notes = list(self.memories.values())
        n = len(notes)
        indices = [int(i) for i in self.retriever.search(query, k)]
        reverse = {}
        if V.has("rev"):
            for src, nt in enumerate(notes):
                for link in nt.links:
                    if valid_link(link, n):
                        reverse.setdefault(int(link), []).append(src)
        order, seen = [], set()
        for i in indices:
            group = [i] + [int(l) for l in notes[i].links if valid_link(l, n)]
            if V.has("rev"):
                group += reverse.get(i, [])
            group = group[:1] + group[1:1 + V.NBR_CAP]      # cap the expansion per retrieved note
            for j in group:
                if V.has("dedupe") and j in seen:
                    continue
                seen.add(j)
                order.append(j)
        parts = []
        for j in order:
            nt = notes[j]
            if V.has("ctxmin"):
                parts.append("talk start time:" + nt.timestamp + "memory content: " + nt.content + "\\n")
            else:
                parts.append(
                    "talk start time:" + nt.timestamp +
                    "memory content: " + nt.content +
                    "memory context: " + nt.context +
                    "memory keywords: " + str(nt.keywords) +
                    "memory tags: " + str(nt.tags) + "\\n"
                )
        return "".join(parts)

'''

MEMORY_PATCHES = [
    ("import V",
     "class RobustBaseLLMController(ABC):",
     "import amem_variants as V\n\n\nclass RobustBaseLLMController(ABC):"),

    ("temperature override and call counter",
     "        def counted(prompt: str, temperature: float = 0.7) -> str:\n"
     "            out = call(prompt, temperature=temperature)\n",
     "        def counted(prompt: str, temperature: float = 0.7) -> str:\n"
     "            temperature = V.effective_temperature(temperature)\n"
     "            V.STATS[\"llm_calls\"] += 1\n"
     "            out = call(prompt, temperature=temperature)\n"),

    ("analysis prompt",
     "        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)",
     "        prompt = V.analyze_prompt(ANALYZE_CONTENT_PROMPT).format(content=content)"),

    ("analysis vetting",
     "            # Final validation\n"
     "            analysis = validate_analysis_result(analysis, content)\n"
     "            return analysis\n",
     "            # Final validation\n"
     "            analysis = validate_analysis_result(analysis, content)\n"
     "            analysis = V.vet_analysis(analysis, content)\n"
     "            return analysis\n"),

    ("retrieval dispatch",
     '        """Find related memories with neighborhood expansion."""\n',
     '        """Find related memories with neighborhood expansion."""\n'
     '        if V.has("dedupe") or V.has("rev") or V.has("ctxmin"):\n'
     '            return self._raw_v2(query, k)\n'),

    ("_raw_v2 method",
     "    # ---- evolution (3 sequential plain-text calls) ----",
     RAW_V2_METHOD + "    # ---- evolution (3 sequential plain-text calls) ----"),

    ("decision counter",
     '            logger.debug("Evolution decision: %s", decision)\n',
     '            logger.debug("Evolution decision: %s", decision)\n'
     '            V.STATS["decision_" + str(decision["decision"])] += 1\n'),

    ("noevo",
     "        neighbor_memory, indices = self.find_related_memories(note.content, k=5)\n",
     '        if V.has("noevo"):\n'
     '            return False, note\n'
     "        neighbor_memory, indices = self.find_related_memories(note.content, k=5)\n"),

    ("noupdate",
     '            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")\n',
     '            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")\n'
     '            if V.has("noupdate"):\n'
     '                should_update = False\n'),

    ("strengthen prompt",
     "strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(",
     "strengthen_prompt = V.strengthen_prompt(STRENGTHEN_DETAILS_PROMPT).format("),

    ("update prompt",
     "update_prompt = UPDATE_NEIGHBORS_PROMPT.format(",
     "update_prompt = V.update_prompt(UPDATE_NEIGHBORS_PROMPT).format("),

    ("connection vetting",
     "                shown = [int(i) for i in indices]\n",
     '                strengthen["connections"] = V.vet_connections(strengthen["connections"])\n'
     "                shown = [int(i) for i in indices]\n"),

    ("strengthen tags",
     '                if strengthen["tags"]:\n'
     '                    note.tags = strengthen["tags"]\n',
     '                new_tags = V.vet_tags(strengthen["tags"], "strengthen_tags")\n'
     '                if new_tags:\n'
     '                    V.log_history(note, "tags", note.tags, new_tags, note.id)\n'
     '                    note.tags = new_tags\n'),

    ("neighbour update",
     '                        if upd["tags"]:\n'
     '                            notetmp.tags = upd["tags"]\n'
     '                        if upd["context"]:\n'
     '                            notetmp.context = upd["context"]\n',
     '                        new_tags = V.vet_tags(upd["tags"], "update_tags")\n'
     '                        new_ctx = V.vet_context(upd["context"], "update_context")\n'
     '                        if new_tags:\n'
     '                            V.log_history(notetmp, "tags", notetmp.tags, new_tags, note.id)\n'
     '                            notetmp.tags = new_tags\n'
     '                        if new_ctx:\n'
     '                            V.log_history(notetmp, "context", notetmp.context, new_ctx, note.id)\n'
     '                            notetmp.context = new_ctx\n'),
]

# ---------------------------------------------------------------------------
# test_advanced_robust.py -> test_v2.py
# ---------------------------------------------------------------------------

DRIVER_PATCHES = [
    ("import memory layer v2",
     "from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem",
     "from memory_layer_robust_v2 import RobustLLMController, RobustAgenticMemorySystem\nimport amem_variants as V"),

    ("import time",
     "import os\nimport json\nimport argparse\n",
     "import os\nimport json\nimport time as _time\nimport argparse\n"),

    ("query = question + keywords",
     "        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)",
     '        query = (question + " " + str(keywords)) if V.has("qplus") else keywords\n'
     "        raw_context = self.retrieve_memory(query, k=self.retrieve_k)"),

    ("evaluate_dataset signature",
     'refresh_index: bool = False, memory_scope: str = "conversation"):',
     'refresh_index: bool = False, memory_scope: str = "conversation",\n'
     "                     only_samples=None, categories=None):"),

    ("log file name",
     '    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{memory_scope}_{timestamp}.log"',
     '    _tag = ("_" + "-".join(sorted(V.FLAGS))) if V.FLAGS else ""\n'
     '    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{memory_scope}{_tag}_{timestamp}.log"'),

    ("cache folder suffix",
     '"cached_memories_robust_{}_{}{}".format(backend, model, "_refresh" if refresh_index else ""),',
     '"cached_memories_robust_{}_{}{}{}".format(backend, model, "_refresh" if refresh_index else "", V.cache_suffix()),'),

    ("categories",
     "    allow_categories = [1, 2, 3, 4, 5]",
     "    allow_categories = categories or [1, 2, 3, 4, 5]"),

    ("build start",
     '            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")',
     '            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")\n'
     '            V.PHASE = "build"\n'
     '            _build_t0, _build_c0 = _time.time(), V.STATS["llm_calls"]'),

    ("build end",
     '            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")\n'
     "        return agent",
     '            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")\n'
     '            eval_logger.info(f"Build stats sample {sample_idx}: {_time.time() - _build_t0:.0f}s, "\n'
     "                             f\"llm_calls={V.STATS['llm_calls'] - _build_c0}, notes={len(memories_to_cache)}\")\n"
     '            V.PHASE = "answer"\n'
     "        return agent"),

    ("read-time variants",
     '            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")',
     '            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")\n'
     "            V.apply_read_variants(agent)"),

    ("sample filter",
     "    for sample_idx, sample in enumerate(samples):\n"
     '        agent = pooled_agent if memory_scope == "pooled" else load_or_build(sample_idx, sample)',
     "    for sample_idx, sample in enumerate(samples):\n"
     "        if only_samples is not None and sample_idx not in only_samples:\n"
     "            continue\n"
     '        agent = pooled_agent if memory_scope == "pooled" else load_or_build(sample_idx, sample)'),

    ("results fields",
     '        "llm_failures": dict(FAILURES),',
     '        "variant": sorted(V.FLAGS),\n'
     '        "variant_stats": dict(V.STATS),\n'
     '        "only_samples": sorted(only_samples) if only_samples else None,\n'
     '        "llm_failures": dict(FAILURES),'),

    ("command-line options",
     "    args = parser.parse_args()",
     '    parser.add_argument("--samples", type=str, default=None,\n'
     '                        help="Comma-separated conversation indices to evaluate (e.g. 1 or 0,2)")\n'
     '    parser.add_argument("--categories", type=str, default=None,\n'
     '                        help="Comma-separated question categories to evaluate (e.g. 3 or 1,2,4)")\n'
     "    args = parser.parse_args()"),

    ("call",
     "        args.refresh_index, args.memory_scope,\n    )",
     "        args.refresh_index, args.memory_scope,\n"
     '        only_samples=({int(x) for x in args.samples.split(",")} if args.samples else None),\n'
     '        categories=([int(x) for x in args.categories.split(",")] if args.categories else None),\n'
     "    )"),
]


def apply(src_name, dst_name, patches):
    text = open(src_name, encoding="utf-8").read()
    problems = []
    for name, old, new in patches:
        n = text.count(old)
        if n != 1:
            problems.append((name, n))
            continue
        text = text.replace(old, new, 1)
    return text, problems


def main():
    jobs = [("memory_layer_robust.py", "memory_layer_robust_v2.py", MEMORY_PATCHES),
            ("test_advanced_robust.py", "test_v2.py", DRIVER_PATCHES)]
    results, failed = [], False
    for src, dst, patches in jobs:
        text, problems = apply(src, dst, patches)
        if problems:
            failed = True
            for name, n in problems:
                print("PATCH FAILED in %s: '%s' (anchor found %d times, expected 1)" % (src, name, n))
        results.append((dst, text))
    if failed:
        print("Nothing was written. Send me the messages above.")
        sys.exit(1)
    for dst, text in results:
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote", dst)
    print("done: %d + %d patches applied" % (len(MEMORY_PATCHES), len(DRIVER_PATCHES)))


if __name__ == "__main__":
    main()
