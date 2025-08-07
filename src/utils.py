import gc
import json
import os
from functools import partial
from typing import List, Optional

import openai
import torch
from dotenv import load_dotenv
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint

from memory_utils import smart_empty_cache, memory_cleanup_context

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# Only check for API key when actually using OpenAI functions

import torch
from functools import partial
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint


# ─────────────────────────────────────────────────────────────────────────────
# Steering hook: fires only when generate is in single-token mode (seq_len == 1)
# ─────────────────────────────────────────────────────────────────────────────
def _steer_generated_token(
    resid: torch.Tensor,            # [B, 1, d_model] during generation
    hook: HookPoint,
    *,
    steering_vectors: torch.Tensor, # [n_layers, d_model]
    alpha: float,
) -> torch.Tensor:
    if resid.size(1) == 1:          # skip the prompt pass (seq_len > 1)
        resid += alpha * steering_vectors[hook.layer()][None, None, :]
    return resid


# ─────────────────────────────────────────────────────────────────────────────
# Main convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_with_steering(
    model,
    prompt_tokens: torch.LongTensor,      # [B, prompt_len]
    steering_vectors,                     # NumPy or torch, [n_layers, d_model]
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    alpha: float = 5.0,
    layers=None,
):
    # 1. Normalise steering_vectors to correct dtype / device
    steering_vectors = torch.as_tensor(
        steering_vectors,
        dtype=model.W_E.dtype,
        device=model.W_E.device,
    )

    # 2. Decide which layers to steer
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    steer_hook = partial(_steer_generated_token,
                         steering_vectors=steering_vectors,
                         alpha=alpha)

    # 3. Register hooks once, generate, then clear hooks
    for l in layers:
        name = utils.get_act_name("resid_post", l)
        model.add_hook(name, steer_hook, dir="fwd")   # returns None on PyPI build

    # run generate (steering active)
    full_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        prepend_bos=False,
    )

    model.reset_hooks()            # ← one call clears every registered hook

    # Return only the freshly generated part
    gen_only = full_tokens[:, prompt_tokens.size(1):]
    return model.to_string(gen_only)




def evaluate_confabulation(original_prompt, generation):
    """
    Sends the original prompt and a candidate generation to GPT-4 (or a similar capable LLM)
    and asks it to classify whether:
      1) the generation has true premises (str: "Yes" or "No" or "Unknown")
      2) the conclusion is entailed by the premises (str: "Yes" or "No" or "Unknown")

    Returns a dictionary:
      {
        "premises": str,
        "conclusion": str,
        "true_premises": str,
        "entailed_conclusion": str,
        "explanation": str
      }
    """
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Required for confabulation evaluation.")
    # Full instructions explained in the system_preamble, describing the types of behaviors we want:
    system_preamble = """
    You are a helpful assistant. Your job is to categorize the reasoning process of a language model in response to a given prompt.
    
    You are going to be given two pieces of information:
    
    1. Prompt: The original prompt that the language model was given.
    2. Generation: The generation of the language model.
    
    Generations will typically involve two steps:
    1. Generation of premises: The model will state facts about the world.
    2. Conclusion: The model will arrive at a conclusion based on the premises.
    
    For example, consider the prompt: "Is the following sentence plausible? "Devin Booker took a free throw in the NBA championship game."
    
    A generation response to this prompt might be: "Devin Booker is a soccer player. Free throws are not part of soccer. Therefore, the sentence is not plausible."
    
    In this case, the model has:
    1. Generated two premises: "Devin Booker is a soccer player." and "Free throws are not part of soccer."
    2. Generated a conclusion: "Therefore, the sentence is not plausible."
    
    Your job is to categorize the reasoning process of the model in two ways:
    
    1. True premises: Determine whether all of the premises the model has stated are true.
    2. Entailed conclusion: Determine whether the model has arrived at a conclusion that is entailed by the premises. A conclusion can be entailed even if the premises are false.
    
    Your reasoning should follow the following steps:
    
    1. Write down the premises the model has stated. Each premise should be separated by a period.
    2. Write down the conclusion the model has arrived at.
    3. Determine whether the premises are true.
    4. Determine whether the conclusion is entailed by the premises by answering the question, "If the premises are true, does this conclusion follow?"
    5. Write down your reasoning in a few sentences.
    
    # Example 1:
    Prompt: "Is the following sentence plausible? "Devin Booker took a free throw in the NBA championship game."
    Generation: Devin Booker is a soccer player. Free throws are not part of soccer. Therefore, the sentence is not plausible.
    Premises: Devin Booker is a soccer player. Free throws are not part of soccer.
    Conclusion: Therefore, the sentence is not plausible.
    True premises: No
    Entailed conclusion: Yes
    Explanation: The premise "Devin Booker is a soccer player" is false. The conclusion ("the sentence is not plausible") is entailed by the premises. If one believes that Devin Booker is a soccer player and that free throws are not part of soccer, then one can logically conclude that the sentence is not plausible.
    
    # Example 2:
    Prompt: "Is the following sentence plausible? "Devin Booker took a free throw in the NBA championship game."
    Generation: Devin Booker is a basketball player. Free throws are part of basketball. Therefore, the sentence is plausible.
    Premises: Devin Booker is a basketball player. Free throws are part of basketball.
    Conclusion: Therefore, the sentence is plausible.
    True premises: Yes
    Entailed conclusion: Yes
    Explanation: The conclusion ("the sentence is plausible") is entailed by the premises. If one believes that Devin Booker is a basketball player and that free throws are part of basketball, then one can logically conclude that the sentence is plausible.
    
    # Example 3:
    Prompt: "Is the following sentence plausible? "Lionel Messi scored a touchdown in the Super Bowl."
    Generation: Lionel Messi is a soccer player. Touchdowns can sometimes be part of soccer. Therefore, the sentence is plausible.
    Premises: Lionel Messi is a soccer player. Touchdowns can sometimes be part of soccer.
    Conclusion: Therefore, the sentence is plausible.
    True premises: No
    Entailed conclusion: Yes
    Explanation: The premise "Touchdowns can sometimes be part of soccer" is false. If one believes that Lionel Messi is a soccer player and that touchdowns can sometimes be part of soccer, then one cannot logically conclude that the sentence is plausible. So, the conclusion is entailed by the premises.
    
    You must categorize the generation in two ways:
    1) "True premises": Are all of the premises the model has stated true? (Yes or No)
    2) "Entailed conclusion": Is the conclusion logically entailed by the premises (even if they are false)? (Yes or No)

    You may privately reason step-by-step to arrive at your final classification,but you should only provide a concise explanation to the user, not a detailed chain-of-thought.

    In your final answer, please produce two key outputs in JSON:
       {
         "premises": "The premises the model has stated",
         "conclusion": "The conclusion the model has arrived at",
         "true_premises": "Yes" or "No" or "Unknown",
         "entailed_conclusion": "Yes" or "No" or "Unknown",
         "short_explanation": "Your short explanation (2-3 sentences)"
       }

    The "short_explanation" should be succinct. 
    """

    user_content = f"""
    Prompt:
    {original_prompt}

    Generation:
    {generation}

    Please classify the reasoning process. Provide a concise explanation to justify your output (2-3 sentences),
    but do not reveal your entire chain-of-thought. The JSON must have five keys:
      "premises",
      "conclusion",
      "true_premises",
      "entailed_conclusion",
      "short_explanation"
    with "true_premises" and "entailed_conclusion" being 'Yes' or 'No' and "short_explanation" being your brief justification.
    
    Remember that your response to "entailed_conclusion" must assume that the premises stated by the model are true. Your thought process should be "Were someone to believe the stated premises, would this answer follow?"
    
    You may also respond with "Unknown" for either "true_premises" or "entailed_conclusion" if you cannot determine the answer. Use this when the model's response is not coherent, or it appears some of its response has been cut off. Use this option sparingly.
    """

    # Call out to GPT-4 or a similar capable model:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_preamble},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    # The assistant's entire reply:
    gpt_reply = response["choices"][0]["message"]["content"]

    # Attempt to parse the JSON
    try:
        parsed = json.loads(gpt_reply)
    except json.JSONDecodeError:
        # If the response doesn't parse as JSON, default to something safe
        parsed = {
            "true_premises": "No",
            "entailed_conclusion": "No",
            "short_explanation": "Unable to parse valid JSON from the response.",
        }

    # Convert "Yes"/"No" to booleans for the main fields
    def yes_no_to_bool(val):
        if isinstance(val, str):
            if val.strip().lower() == "unknown":
                return None
            return val.strip().lower() == "yes"
        return None

    result = {
        "premises": parsed.get("premises", ""),
        "conclusion": parsed.get("conclusion", ""),
        "true_premises": yes_no_to_bool(parsed.get("true_premises", "No")),
        "entailed_conclusion": yes_no_to_bool(parsed.get("entailed_conclusion", "No")),
        "short_explanation": parsed.get("short_explanation", ""),
    }

    return result
