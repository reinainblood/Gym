# FACTS Search protocol

This implementation follows the FACTS Search V2 leaderboard and version 3 of the linked Search-On notebook. Search-On
is the only scored method on the public leaderboard. Search-Off is used during dataset construction and adversarial
filtering; it is not a leaderboard evaluation method.

The Search-On protocol uses:

- the published system prompt;
- Brave Web Search with five results per query;
- one or more parallel searches in a model turn;
- at most seven search turns;
- the published tool-free final-answer request when the model is still searching after turn seven;
- one short final answer;
- Gemini 3.5 Flash with the published A/B/C grading prompt;
- F1, accuracy, attempted accuracy, hedging rate, average search turns, and a 1,000-resample bootstrap interval.

## Public data

The V2 leaderboard describes 1,842 questions: 921 public and 921 private. The linked downloadable dataset is version 1
and contains 890 public rows. The preparation script verifies the CSV checksum and row count before writing the Gym
dataset. It fails if either value changes.

The 31 additional public questions and all 921 private questions are unavailable. The CSV contains `example_id`,
`problem`, and `gold answer`, but no subset labels, so this adapter does not report Hard Tail, Wiki Two-hop, Wiki
Multi-hop, or KG Hops breakdowns.

The V2 leaderboard names Gemini 3.5 Flash as the grader. The linked notebook still names the retired Gemini 2.0 Flash.
This adapter uses Gemini 3.5 Flash with the notebook's grading prompt.
