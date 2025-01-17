from functools import partial
from typing import Optional, List
import torch
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
import re
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import openai
import json
import os

openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError('OPENAI_API_KEY not found in environment variables')

def steer_residual_stream(
    residual_component: torch.FloatTensor,
    hook: HookPoint,
    steering_vectors: torch.Tensor,
    alpha: int = 5,
    instruction_pos: int = 0,
) -> torch.FloatTensor:
    """
    Steer the residual stream by adding a scaled steering vector only to positions after instruction_pos.

    Args:
        residual_component: The current residual activation (batch_size, seq_len, d_model).
        hook: The HookPoint.
        steering_vectors: Pre-computed steering vectors for each layer.
        alpha: Scaling factor for the steering vector.
        instruction_pos: Token position after which to apply the steering.

    Returns:
        Modified residual component after steering.
    """
    steering_vector = steering_vectors[hook.layer()]  # Shape: (d_model,)
    add_act = torch.tensor(alpha * steering_vector).to(residual_component.device)

    # Apply steering only to positions after the instruction
    batch_size, seq_len, _ = residual_component.shape
    if seq_len > instruction_pos:
        residual_component[:, instruction_pos:, :] += add_act

    add_act.detach_()
    return residual_component

def generate_with_hooks(
    model,
    tokens: torch.Tensor,
    steering_vectors: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    verbose: bool = True,
    alpha: float = 5.0,
    layers: Optional[List[int]] = None,
) -> str:
    """
    Generate text while steering the residual stream (through hooks)
    AND benefiting from key-value caching to avoid re-running the full prompt.

    The steering function (steer_residual_stream) only applies additions to residual
    positions strictly beyond the initial `instruction_pos`.
    """

    # --------------------------------------------------------------------------
    # 1) Initialize a key-value cache for the model
    #    So we can reuse it at each decoding step.
    # --------------------------------------------------------------------------
    kv_cache = HookedTransformerKeyValueCache.init_cache(
        cfg=model.cfg,
        device=tokens.device,
        batch_size=tokens.size(0),
    )

    # --------------------------------------------------------------------------
    # 2) Figure out which layers we want to steer
    # --------------------------------------------------------------------------
    # If no layers are specified, we steer all layers
    if layers is None:
        layers = range(model.cfg.n_layers)

    # The "instruction_pos" is the boundary token index beyond which
    # we apply the steering
    instruction_pos = tokens.size(1)

    # --------------------------------------------------------------------------
    # 3) Build our hook function that adds the "steering_vectors" in the
    #    residual stream after `instruction_pos`.
    # --------------------------------------------------------------------------
    partial_steer_func = partial(
        steer_residual_stream,
        steering_vectors=steering_vectors,
        alpha=alpha,
        instruction_pos=instruction_pos,
    )

    # Each layer's "resid_post" will register the same partial function
    # (or different ones, if needed).
    hooks = [
        (utils.get_act_name("resid_post", layer), partial_steer_func)
        for layer in layers
    ]

    # --------------------------------------------------------------------------
    # 4) First forward pass over the entire prompt to:
    #    (a) fill the kv_cache
    #    (b) retrieve logits for the final prompt token
    # --------------------------------------------------------------------------
    with torch.no_grad():
        logits_full_prompt = model.run_with_hooks(
            tokens,
            fwd_hooks=hooks,
            return_type="logits",
            past_kv_cache=kv_cache,   # This populates kv_cache with the entire prompt
        )  # shape: [batch, seq_len, vocab_size]

    # Use the final token's logits if you want to sample the first new token
    # but in practice we'll do that inside the loop below
    model.reset_hooks()  # Clear ephemeral hooks before next step

    generated_tokens = []

    # --------------------------------------------------------------------------
    # 5) Generate new tokens, one step at a time, reusing kv_cache
    # --------------------------------------------------------------------------
    for _ in range(max_new_tokens):
        # Only run forward on the last token we appended
        with torch.no_grad():
            logits_step = model.run_with_hooks(
                tokens[:, -1:],
                fwd_hooks=hooks,
                return_type="logits",
                past_kv_cache=kv_cache,  # Reuse & update the same cache
            )
        model.reset_hooks()

        # logits_step shape: [batch, 1, vocab_size]
        next_logits = logits_step[:, -1, :]
        # Apply temperature
        next_logits = next_logits / temperature
        probs = torch.nn.functional.softmax(next_logits, dim=-1)

        # Sample next token. (Assuming batch_size = 1 for simplicity)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token_id)

        # Append new token to the existing tokens
        next_token_tensor = torch.tensor([[next_token_id]], device=tokens.device)
        tokens = torch.cat([tokens, next_token_tensor], dim=1)

        # Optional Stop Conditions
        if next_token_id == model.tokenizer.eos_token_id:
            break

        # Print the decoded token if desired
        if verbose:
            print(model.tokenizer.decode([next_token_id], skip_special_tokens=False), end='')

    # --------------------------------------------------------------------------
    # 6) Decode the newly added tokens into text
    # --------------------------------------------------------------------------
    generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    if verbose:
        print()

    return generated_text

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
            {"role": "user", "content": user_content}
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
            "short_explanation": "Unable to parse valid JSON from the response."
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
        "short_explanation": parsed.get("short_explanation", "")
    }

    return result
