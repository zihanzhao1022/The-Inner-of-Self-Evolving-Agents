"""Per-model prompt wrapping strategies for generation-time evaluation.

We need to wrap a raw user prompt in the model's training-distribution
template so the generation reflects its in-distribution behaviour. Three
template families cover the 9 models we evaluate:

  - "raw"          : no wrapping at all. The prompt is passed verbatim.
                     Used for base / continued-pretraining models
                     (Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-Coder-3B,
                     Qwen2.5-Coder-7B) which were never trained on a
                     chat template. The model treats the prompt as a
                     text-completion task — that IS its native behaviour.

  - "qwen_chatml"  : full official Qwen ChatML, fetched via
                     tokenizer.apply_chat_template, with the default
                     system prompt "You are a helpful assistant." Used
                     for the Instruct models (Qwen2.5-3B-Instruct,
                     Qwen2.5-7B-Instruct).

  - "azr_r1"       : the DeepSeek-R1-style instruction_following template
                     copied verbatim from
                     https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
                     /blob/master/absolute_zero_reasoner/data_construction
                     /process_data.py. Ends with the literal prefix
                     "Assistant: <think>" which forces R1-trained models
                     into CoT generation. Used for the three AZR variants
                     (AZR-Coder-3B, AZR-Base-7B, AZR-Coder-7B).
"""

from __future__ import annotations


# Copied verbatim from AZR upstream:
#   absolute_zero_reasoner/data_construction/process_data.py
# Note: AZR also overrides tokenizer.chat_template to plain concat in
# main_azr_ppo.py, so the model sees this raw text (no <|im_start|> tokens).
AZR_R1_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think> "
    "</think> and <answer> </answer> tags, respectively, i.e., <think> "
    "reasoning process here </think> <answer> answer here </answer>. "
    "User: {prompt}\nAssistant: <think>"
)


def wrap_raw(prompt: str, **_kwargs) -> str:
    """No-template completion — feed prompt verbatim. Native for base models."""
    return prompt


def wrap_azr_r1(prompt: str, **_kwargs) -> str:
    """AZR's training-distribution prompt. Ends with 'Assistant: <think>'."""
    return AZR_R1_TEMPLATE.format(prompt=prompt)


def wrap_qwen_chatml(prompt: str, tokenizer=None, **_kwargs) -> str:
    """Full official Qwen ChatML, via tokenizer.apply_chat_template.

    Includes the default system prompt 'You are a helpful assistant.' and
    the <|im_start|>assistant\\n generation suffix.

    Requires a tokenizer argument so the model-specific template is used
    (Qwen2.5-7B-Instruct vs Qwen2.5-3B-Instruct may diverge in the future).
    """
    if tokenizer is None:
        raise ValueError("wrap_qwen_chatml requires tokenizer=...")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


WRAPPERS = {
    "raw":         wrap_raw,
    "qwen_chatml": wrap_qwen_chatml,
    "azr_r1":      wrap_azr_r1,
}


def wrap_prompt(prompt: str, mode: str, tokenizer=None) -> str:
    """Dispatch to the named wrapper. Raises if mode is unknown."""
    if mode not in WRAPPERS:
        raise ValueError(f"unknown template mode {mode!r}; "
                         f"choose from {sorted(WRAPPERS)}")
    return WRAPPERS[mode](prompt, tokenizer=tokenizer)
