# FACTS Search agent

The agent implements the public FACTS Search-On loop. A search hop is one model turn containing one or more parallel
`brave_search` calls. The model may use up to seven search hops. If the seventh hop contains tool calls, the agent
executes them and sends the published tool-free final request: `Please provide a final answer based on the information
gathered so far.`

The Gym rollout records the model and tool trajectory, search-hop count, query count, and whether the final tool-free
request was needed.
