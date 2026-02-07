# RL Training Integration

This guide covers integrating `strands-env` with RL training frameworks, specifically [slime](https://github.com/THUDM/slime/).

## Overview

`strands-env` captures token-level observations (TITO) during agent rollouts, which are essential for on-policy RL training. The `TokenObservation` contains:
- `token_ids` - All tokens (prompt + rollout)
- `rollout_token_ids` - Only generated tokens
- `rollout_logprobs` - Log probabilities for each generated token
- `rollout_loss_mask` - Mask for loss computation

## Slime Integration

Customize the `generate` and `reward_func` methods to replace single generation with agentic rollout:

```python
from strands_env.core import Action, TaskContext
from strands_env.core.models import sglang_model_factory
from strands_env.utils import get_cached_client_from_slime_args

async def generate(args, sample, sampling_params):
    # Build model factory with cached client
    factory = sglang_model_factory(
        model_id=args.hf_checkpoint,
        tokenizer=tokenizer,
        client=get_cached_client_from_slime_args(args),
        sampling_params=sampling_params,
    )

    # Create environment and run step
    env = YourEnv(model_factory=factory, reward_fn=None)
    action = Action(message=sample.prompt, task_context=TaskContext(ground_truth=sample.label))
    step_result = await env.step(action)

    # Extract TITO data for training
    token_obs = step_result.observation.tokens
    sample.tokens = token_obs.token_ids
    sample.loss_mask = token_obs.rollout_loss_mask
    sample.rollout_log_probs = token_obs.rollout_logprobs
    sample.response_length = len(token_obs.rollout_token_ids)

    # Attach for reward computation
    sample.action = action
    sample.step_result = step_result
    return sample

async def reward_func(args, sample, **kwargs):
    reward_fn = YourRewardFunction()
    reward_result = await reward_fn.compute(action=sample.action, step_result=sample.step_result)
    return reward_result.reward
```

## Key Points

- **Connection pooling**: `get_cached_client_from_slime_args(args)` provides connection pooling across rollouts for efficient GPU utilization
- **Token observations**: `TokenObservation` contains token IDs and logprobs for on-policy training (SGLang backend only)
- **Async rewards**: Reward is computed separately to allow async/batched reward computation
- **Model factory pattern**: Each `step()` creates a fresh model instance for clean token tracking state
