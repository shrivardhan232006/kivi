Today most of the voice dictation tools that exist out there are good at recognizing,correcting speech and saving them as history but none of them work like personalized agents which would really make a user's work more efficient and fast. There needs to be some way to efficiently use the history thats getting saved and build up on that memory to keep track of all the work that the user is doing and build a semantic and personalized memory around it. That's why I built Kivi.

Kivi turns your voice dictations into a searchable personal memory.Kivi solves this by using LLMs to extract structured knowledge from your dictation history: people, projects, preferences, and events.

Why does using semantic memory make kivi special?
Semantic memory is something that humans use to store and retrieve factual information. It helps have a human to have a personalized assistant who can keep track of all the work, there personal life and other things that requires attention. It creates a working memory for a user.It creates value in:
1.You don't have to redictate the same task everytime.
2.You don't have to remember everything yourself
3.A system that only speaks based on your history and doesn't hallucinate on unknown terms
4.It can remember you preferences ,your relationships and your projects.
5.It recognize your tone and replicate it at different tasks reducing your work of dictation.

Kivi's Architecture:

```
                              [ User Dictation ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 1. Dictionary Correction  │ (ASR noise repair, deduplication)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 2. History Storage (SQL)  │ (Immutable dictation archive)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 3. LLM Memory Extractor   │ (Factual, Episodic, Preference)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 4. Conflict Resolution    │ (Bi-temporal invalidation / update)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 5. Dual Vector Store      │ (History & Memory embeddings)
                        └───────────────────────────┘
                                      │
======================================│======================================
                          QUERY & INFERENCE PIPELINE
======================================│======================================
                                      │
                                [ User Query ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Contextualization Layer   │ (Rewrites follow-up questions)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Fast-Path Check           │ (Instant bypass for direct facts)
                        └───────┬───────────┬───────┘
                                │ Bypass    │ Fallback / Complex
                                ▼           ▼
                        ┌──────────┐   ┌───────────────────────────┐
                        │ Direct   │   │ LLM Function Router       │
                        │ Lookup   │   └─────────────┬─────────────┘
                        └────┬─────┘                 │
                             │   ┌───────────────────┴───────────────────┐
                             │   │                   │                   │
                             ▼   ▼                   ▼                   ▼
                     ┌───────────────┐       ┌───────────────┐   ┌───────────────┐
                     │ lookup_fact() │       │  aggregate()  │   │ get_valid_at()│
                     └───────┬───────┘       └───────┬───────┘   └───────┬───────┘
                             │                       │                   │
                             └───────────────┬───────┴───────────────────┘
                                             │ (If 0 results: Fallback)
                                             ▼
                                     ┌───────────────┐
                                     │ fuzzy_search()│ (Semantic + BM25 + Recency)
                                     └───────┬───────┘
                                             │
                                             ▼
                        ┌───────────────────────────┐
                        │ Context Formatter         │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ LLM Answer Generator      │ (Drafts persona-aligned response)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Grounding Verifier Pass   │ (Flags hallucinations / unsupported)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        [ Verified Answer + Provenance Links ]
```

1.The LLM chooses _which tool_ to call and with _what arguments_, but the retrieval itself is executed by deterministic, reproducible SQL queries and three-signal algorithms.
2.Every memory tracks both transaction time (`created_at`) and valid time (`valid_from`, `valid_to`). Facts that change over time (such as office moves or project renames) supersede older records without deleting history.
3.Queries asking for complete sets (e.g. "Who are all the people I talk to on Slack?") use `SELECT DISTINCT` SQL aggregations rather than vector similarity top-K ranking, guaranteeing zero omissions.
4.For fuzzy and episodic questions, Kivi fuses **Semantic Embeddings** (45%), **BM25 Lexical Matching** (35%), and **Timeline Recency Decay** (20%).
5.Verification: Every generated answer undergoes an automated verification pass before reaching the user. Any claim not strictly supported by the source dictation context is flagged or abstained from.

Where This Goes
Right now Kivi build up in batches. The are some little bugs with the retrieving and holding up context or with the semantic meaning. Further I would like to improve it by making it and builduing the memory real time not in batches and Kivi surfacing relevant context before you even ask. The memory layer is already built for it. The bi-temporal schema handles updates. The infrastructure is present it just needs the live pipe.

The vision is simple: you speak, Kivi remembers, and when you need it back, you just ask.
