## Empath therapeutic reasoning kernel

This repository contains a small Python miniKanren kernel for coaching-oriented
ACT/CBT/REBT/DBT reasoning. It is intended to produce inspectable therapeutic
hypotheses and intervention candidates, not diagnoses.

```python
from empath import CoachingState, TherapeuticReasoningKernel

kernel = TherapeuticReasoningKernel()
kernel.add_state(
    CoachingState(
        state_id="prototype",
        thoughts=("If it is bad, it proves I am not cut out for this.",),
        emotions=("anxiety", "shame"),
        behaviors=("avoidance",),
        values=("mastery", "autonomy"),
    )
)

print(kernel.hypotheses_for("prototype"))
print(kernel.ranked_interventions("prototype", limit=3))
print(kernel.states_for_intervention("cognitive_defusion"))
```

Relational queries can also run "backwards" over known states:

```python
print(kernel.patterns_for_intervention("cognitive_defusion"))
print(kernel.states_for_pattern("fusion", source="act"))
print(kernel.safe_states_for_intervention("rebt_disputation"))
print(kernel.contraindicated_states_for_intervention("rebt_disputation"))
print(kernel.compare_intervention_across_states("rebt_disputation"))
```

Architecture:

```text
structured observations -> miniKanren hypotheses -> safe candidates -> ranking
```

The chat app also keeps a longitudinal working formulation per session:

```text
turn extraction + kernel hypotheses + chosen intervention -> formulation graph
```

The graph stores tentative observations, hypotheses, interventions, and
relationships with turn provenance. The browser exposes this as a Working Map
beside the trace inspector, and each node can be marked as fitting, not quite,
or removed. User corrections persist in the in-memory session and removed nodes
are not reintroduced by later turns.

Each assistant turn also proposes a tiny N-of-1 coaching experiment. The
experiment layer takes the selected intervention, supporting hypotheses, and
working-map delta, then turns them into a small action, prediction, measure,
and timebox. The browser shows the experiment under the assistant turn and lets
the user mark whether it helped, did not help, was too hard, or was skipped.

The relational layer answers questions like:

- what CBT/REBT/ACT/DBT patterns may be present?
- what broader coaching focus is active?
- what interventions are logically coherent for this state?
- what known states would justify a given intervention?
- which candidates are contraindicated before validation or safety handling?

Therapeutic systems are now modular. ACT, CBT, REBT, DBT, coaching focus, and
cross-system loops live under `empath/therapeutic_systems/`, and
`TherapeuticReasoningKernel` accepts an optional `systems=` tuple. A new system,
such as focusing, can register its own pattern relation, intervention mappings,
exercises, modalities, and ranking bonus without changing the coordinator
kernel.

The DBT module currently targets:

- mindfulness / observe-and-describe
- distress tolerance / crisis-survival pause
- emotion regulation / check-the-facts style mapping
- interpersonal effectiveness / ask-or-boundary scripting
- self-validation when the user appears to invalidate their own feeling

The coaching focus module maps observations into 12 broader focus areas:

- values, purpose, and direction
- goal setting and behavioral activation
- motivation, willingness, and persistence
- cognitive patterns and belief change
- emotional regulation and distress tolerance
- avoidance, procrastination, and experiential escape
- self-concept, confidence, and self-efficacy
- decision-making and problem solving
- interpersonal effectiveness and boundaries
- attention, focus, and environment design
- resilience, relapse prevention, and recovery
- integration and review

These focus moves use a low ranking bonus so they enrich traces and the Working
Map without displacing more specific ACT/CBT/REBT/DBT interventions.

The default loop system currently recognizes:

- avoidance plus identity threat
- sadness/anxiety minimal disclosure
- shame/self-worth fusion
- procrastination around a concrete valued action
- high-distress gating

Run the demo:

```bash
uv run python hello.py
```

Run the DeepSeek-backed chat CLI:

```bash
uv run python -m empath.chat
```

Run the API and SSE chat app:

```bash
uv run python -m empath.api
```

Then open `http://127.0.0.1:8000`.

For local testing without DeepSeek calls:

```bash
uv run python -m empath.api --dry-run
```

Storage backend:

The API keeps hot workspace objects in memory and persists a serialized
workspace snapshot through a pluggable backend. The default CLI path uses a
local SurrealDB file at `.empath_surreal.db`; pass `--store-backend memory` for
throwaway sessions or `--store-backend json --state-file ...` for the legacy
JSON snapshot backend.

```bash
uv run python -m empath.api --store-backend memory
uv run python -m empath.api --store-backend json --state-file .empath_chat_state.json
uv run python -m empath.api --store-backend surreal --surreal-url mem://
```

To point the default SurrealDB backend at a running SurrealDB service:

```bash
uv run python -m empath.api \
  --store-backend surreal \
  --surreal-url ws://127.0.0.1:8000/rpc \
  --surreal-user root \
  --surreal-password root
```

The SurrealDB backend keeps the full app snapshot as a recovery record and also
projects queryable records into:

- `empath_user`
- `empath_workspace`
- `empath_conversation`
- `empath_message`
- `working_node`
- `working_edge`
- `working_node_provenance`
- `working_edge_provenance`
- `coaching_experiment`
- `working_compaction_policy`

The projection is rebuilt on each snapshot save, so query results stay aligned
with the current workspace state while the Python snapshot remains the source
of truth for app restoration. `working_node` and `working_edge` records include
database-visible compaction fields such as `active_in_map`,
`hidden_by_policy`, `protected_by_policy`, `compaction_reason`,
`retention_action`, `age`, and `priority_score`.

API surface:

- `GET /`: browser chat app
- `GET /api/health`: service metadata
- `GET /api/chat/session?session_id=...`: visible multi-turn transcript
- `GET /api/chat/explain?session_id=...&message_index=...`: lazy readable rationale for an assistant turn
- `POST /api/chat`: one-shot JSON chat turn
- `GET /api/chat/stream?session_id=...&message=...&trace=1`: SSE chat turn
- `GET /api/formulation?session_id=...`: current working formulation graph
- `GET /api/formulation/compaction?workspace_id=...`: database compaction policy summary
- `GET /api/formulation/mirror?session_id=...`: reflective playback of the working formulation
- `POST /api/formulation/feedback`: confirm, reject, or remove a formulation node
- `GET /api/experiments?session_id=...`: proposed coaching experiments and outcomes
- `POST /api/experiments/feedback`: close the loop on one experiment

The API keeps an in-memory transcript per `session_id`; the browser app stores
the active session id in local storage and reloads the transcript when the page
opens. Assistant messages expose a `Why this?` chip when a trace is available;
the rationale is generated only after that chip is clicked. SSE chat turns emit
a `formulation` event with the latest graph delta after the response plan is
selected and an `experiment` event with the proposed learning-loop action. The
Working Map can also be mirrored back using reflective listening; live mode
uses the model to verbalize the graph, while dry-run mode uses the deterministic
graph playback.

Useful CLI options:

```bash
uv run python -m empath.chat --once "I keep avoiding the prototype because if it is bad, I am a failure."
uv run python -m empath.chat --show-extraction --show-kernel --show-plan
uv run python -m empath.chat --trace
uv run python -m empath.chat --trace-prompts
uv run python -m empath.chat --show-kernel
uv run python -m empath.chat --dry-run --once "I should be able to handle this, but I keep putting it off."
```

The CLI reads the DeepSeek API key from `.deepseek_api_key` by default and uses
the DeepSeek model id `deepseek-v4-flash`.

The live chat path now makes two Pydantic AI calls per user turn:

```text
user message -> structured extraction agent -> miniKanren kernel -> structured response-plan agent -> renderer
```

`--dry-run` does not call DeepSeek. It uses the deterministic fallback extractor
only to preview the prompt shape and kernel output.

Interactive debug commands:

- `/trace`: print the previous turn's full trace
- `/debug`: toggle trace output after every turn
- `/prompts`: toggle prompt inclusion inside traces

Run tests:

```bash
uv run python -m unittest
```

Run the offline kernel eval suite:

```bash
uv run python -m empath.evals
```

The eval suite currently contains 37 fixtures. Each fixture can assert expected
hypotheses, forbidden hypotheses, contraindicated-but-coherent interventions,
and acceptable top-ranked candidates.
