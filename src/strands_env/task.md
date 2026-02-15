Help me onboarding the synthetic agentic environment from https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K to the current environemnt implementation.
You can find examples in [code_sandbox](../../../../../../Users/lyichuan/PycharmProjects/strands-env/src/strands_env/environments/code_sandbox)[code_sandbox](../code_sandbox) and [terminal_bench](../../../../../../Users/lyichuan/PycharmProjects/strands-env/src/strands_env/environments/terminal_bench). 
The data for AgentWorldModel-1K can be found in [data](../../../../data)[data](../../../../../../Users/lyichuan/PycharmProjects/strands-env/data). 

The description of AgentWorldModel-1K can be found in [README.md](../../../../data/AgentWorldModel-1K/README.md)[README.md](../../../../../../Users/lyichuan/PycharmProjects/strands-env/data/AgentWorldModel-1K/README.md).
I want the environment can initiate the sqlite connection based on [gen_db.jsonl](../../../../../../Users/lyichuan/PycharmProjects/strands-env/data/AgentWorldModel-1K/gen_db.jsonl) and [gen_sample.jsonl](../../../../../../Users/lyichuan/PycharmProjects/strands-env/data/AgentWorldModel-1K/gen_sample.jsonl).

There are s

Can you first implement the environment into environments and then tooling then rewards. 

