# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAPO reward manager with remote (process-based) reward computation.

Combines RemoteRewardManager (process-safe math_verify) with DAPO's overlong
buffer penalty. Each compute_score call runs in a dedicated Ray actor process,
so signal.alarm-based timeouts work correctly.
"""

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.remote import RemoteRewardManager


@register("dapo_remote")
class DAPORemoteRewardManager(RemoteRewardManager):

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)

        self.num_examine = config.reward.get("reward_kwargs", {}).get("num_examine", 0)
        self._examine_count = 0

        overlong_buffer_cfg = config.reward.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    async def run_single(self, data: DataProto) -> dict:
        # Get valid_response_length before delegating to parent
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

        # Run reward computation in remote process (parent's run_single)
        result = await super().run_single(data)

        reward = result["reward_score"]
        reward_extra_info = result["reward_extra_info"]

        # Apply DAPO overlong buffer penalty
        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        # Log sample responses for debugging
        if self.num_examine > 0 and self._examine_count < self.num_examine:
            self._examine_count += 1
            import logging
            logger = logging.getLogger("dapo_remote")
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            logger.warning(
                f"[examine {self._examine_count}/{self.num_examine}] "
                f"resp_len={int(valid_response_length)} reward={reward:.3f}\n"
                f"[prompt] {prompt_str[:200]}\n"
                f"[response] {response_str[:500]}\n"
                f"[ground_truth] {ground_truth}"
            )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
