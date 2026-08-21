"""Prompt templates for the QUD pipeline.

Three stages, each a separate LLM call with strict JSON output:

  Stage 1  RECONSTRUCT : answer-only -> addressed question(s) + speech act.
           The asked question is deliberately withheld to prevent the
           model from anchoring on it ("inverse QUD reconstruction").
  Stage 2  RELATE      : asked question vs each reconstructed question ->
           structured relation (equivalent / specification /
           generalization / topic_shift / unrelated).
  Stage 2b COMMITMENT  : does the answer commit to addressing the asked
           question, independent of topical overlap?
  Stage 3  DIRECTNESS  : for equivalent-relation cases only, decide
           Explicit vs Implicit (is the information stated in the
           requested form?).

All templates ask for JSON only. Parsing utilities live in
qud/reconstruct.py and qud/relations.py.
"""

RECONSTRUCT_SYSTEM = """\
You analyze transcripts of political interviews. You will see ONLY the \
respondent's answer, not the interviewer's question. Your job is to infer \
which question or questions this answer would be a DIRECT answer to, based \
solely on the information the answer actually provides.

Rules:
- Write each addressed question as a single, specific interrogative sentence.
- Only include questions the answer genuinely resolves or substantively \
addresses. Do not guess what the interviewer might have asked.
- If the answer provides no information and instead refuses, claims not to \
know, or asks for clarification, return an empty question list and set \
"speech_act" accordingly.
- Return at most {max_quds} questions, most central first.

Respond with JSON only, in this schema:
{{"addressed_questions": ["...", "..."],
  "speech_act": "answer" | "decline" | "ignorance" | "clarify",
  "evidence": "short quote or paraphrase of the answer span supporting q1"}}"""

RECONSTRUCT_USER = """\
Answer given by the respondent:
\"\"\"{answer}\"\"\"

JSON:"""


RELATE_SYSTEM = """\
You compare two questions and decide the relation of question B (what was \
actually addressed) to question A (what was asked).

Definitions:
- "equivalent": B requests the same information as A.
- "specification": B is a narrower sub-question of A; answering B answers \
only one facet or special case of A.
- "generalization": B is a broader, less specific question than A; \
answering B gives only generic information relative to A.
- "topic_shift": B is related to A's general theme but is about a different \
subject, agent, event, or time frame.
- "unrelated": B shares no substantive content with A.

Also output "overlap", a number from 0.0 to 1.0: the fraction of the \
information requested by A that an answer to B would provide.

Respond with JSON only:
{"relation": "...", "overlap": 0.0, "rationale": "one short sentence"}"""

RELATE_USER = """\
Question A (asked by the interviewer):
\"\"\"{asked}\"\"\"

Question B (what the answer actually addressed):
\"\"\"{addressed}\"\"\"

JSON:"""


DIRECTNESS_SYSTEM = """\
You will see an interviewer's question and a respondent's answer that DOES \
address the question. Decide whether the requested information is stated \
EXPLICITLY (in the requested form: a yes/no question gets a yes/no, a \
when-question gets a time, etc.) or only IMPLICITLY (the information can be \
inferred but is never stated in the requested form).

Respond with JSON only:
{"directness": "explicit" | "implicit", "rationale": "one short sentence"}"""

DIRECTNESS_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""


# Direct prompting baseline (the strategy class that dominated the shared
# task; we reimplement it as a comparison system, NOT as our contribution).
DIRECT_BASELINE_SYSTEM = """\
You classify how a politician's answer relates to the question asked, using \
this taxonomy:

Clear Reply
  - Explicit: the requested information is stated in the requested form.
Ambivalent
  - Implicit: the information is inferable but never stated directly.
  - Dodging: the answer ignores the question entirely.
  - Deflection: the answer shifts to a different topic, person, or question.
  - General: the answer is too vague/non-specific to resolve the question.
  - Partial/half-answer: only one facet of the question is answered.
Clear Non-Reply
  - Declining to answer: the respondent refuses to answer.
  - Claims ignorance: the respondent says they do not know.
  - Clarification: the respondent asks for clarification instead.

Think step by step, then respond with JSON only:
{"reasoning": "...", "evasion_label": "<one of the 9 labels above>"}"""

DIRECT_BASELINE_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""


# ===========================================================================
# COMMITMENT (Stage 2b): does the answer COMMIT to addressing the question,
# independent of topical overlap? Topical relevance (the RELATE overlap) and
# answer-commitment are orthogonal: a vague/General evasion is fully on-topic
# yet commits to nothing. This judgment is made on the ORIGINAL asked question
# and the answer directly (not the reconstructed QUD), so it is robust to
# reconstruction drift.
# ===========================================================================

COMMITMENT_SYSTEM = """\
You see an interviewer's question and a respondent's answer. Judge how much \
the answer COMMITS to actually providing what the question asks for. This is \
NOT about topic relevance: an answer can be entirely on-topic yet commit to \
nothing (vague, hedged, or non-specific).

Use exactly one label:
- "full": the answer provides the requested information or takes a clear, \
specific position on what was asked.
- "partial": the answer provides some of the requested information but hedges, \
omits key parts, or only addresses one facet.
- "evasive": the answer stays on the topic of the question but avoids \
committing — it is vague, generic, deflects to a related matter, or talks \
around the point without resolving it.
- "none": the answer does not engage the question's content at all (refuses, \
claims ignorance, changes subject entirely, or asks for clarification).

Also output "commitment", a number from 0.0 (no commitment) to 1.0 (fully \
committed), consistent with the label.

Respond with JSON only:
{"commitment_label": "full" | "partial" | "evasive" | "none",
 "commitment": 0.0,
 "rationale": "one short sentence"}"""

COMMITMENT_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""

# ===========================================================================
# HIERARCHICAL-TAXONOMY CoT (prompt-only variant of the direct baseline).
# Reasons down the taxonomy tree: first the coarse clarity branch
# (reply / ambivalent / non-reply), then the fine-grained strategy WITHIN
# that branch. This "cognitive scaffolding" mirrors the taxonomy hierarchy
# used to derive clarity from evasion, and is reported to help reasoning
# models on this task. Append few-shot examples to SYSTEM exactly as the
# flat baseline does.
# ===========================================================================

DIRECT_HIER_SYSTEM = """\
You classify how a politician's answer responds to the question asked. Work \
through the taxonomy in TWO stages and think step by step.

STAGE 1 - Decide the clarity branch:
- "Clear Reply": the answer provides the requested information.
- "Ambivalent": the answer engages the topic but does not clearly provide \
the requested information (vague, partial, deflected, or only inferable).
- "Clear Non-Reply": the answer does not engage the question's content at all \
(refuses, claims ignorance, or asks for clarification).

STAGE 2 - Pick the fine-grained strategy WITHIN the chosen branch:
- If Clear Reply -> "Explicit" (stated in the requested form).
- If Ambivalent -> one of:
    "Implicit"  (the information is inferable but never stated directly),
    "Dodging"   (ignores the question entirely while still talking),
    "Deflection"(shifts to a different topic, person, or question),
    "General"   (too vague/non-specific to resolve the question),
    "Partial/half-answer" (answers only one facet of the question).
- If Clear Non-Reply -> one of:
    "Declining to answer" (explicitly refuses),
    "Claims ignorance"    (says they do not know),
    "Clarification"       (asks for clarification instead of answering).

Reason briefly through Stage 1 then Stage 2, then output JSON only:
{"clarity_branch": "Clear Reply" | "Ambivalent" | "Clear Non-Reply",
 "reasoning": "one or two short sentences",
 "evasion_label": "<one of the 9 fine-grained labels above>"}"""

DIRECT_HIER_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""