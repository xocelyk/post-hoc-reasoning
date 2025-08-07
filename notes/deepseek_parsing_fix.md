# DeepSeek Model Parsing Fix

## Issue
DeepSeek models (DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Llama-8B) were producing responses in a different format than expected by the standard parser. The responses included:

1. The entire prompt with special tokens like `<｜begin▁of▁sentence｜>`, `<｜User｜>`, etc.
2. The actual answer at the end in format: `**Answer:** (A) Yes, contains anachronistic elements.`

This caused parsing failures because the standard parser looked for patterns like "the best answer is:" which DeepSeek models don't use.

## Solution
Created a custom DeepSeek-specific parser (`parse_deepseek_response`) that:

1. **Finds the last occurrence of (A) or (B)** in the response
   - Uses regex pattern: `r'\(\s*([AaBb])\s*\)'`
   - Takes the last match to avoid false positives from examples in the prompt

2. **Maps the letter to yes/no** by:
   - Examining context around the letter (50 chars before/after)
   - Looking for keywords like "yes", "no", "contains anachronistic", etc.
   - Using a fallback heuristic if context is unclear

3. **Integration** in `nnsight_utils/experiment_runner.py`:
   - Modified `parse_response` method to accept optional `model_name` parameter
   - Automatically uses DeepSeek parser when model name starts with 'deepseek'
   - Updated all calls to pass the model name

## Files Modified
1. `src/parsing_utils.py` - Added `parse_deepseek_response` function
2. `src/nnsight_utils/experiment_runner.py` - Updated to use DeepSeek parser for DeepSeek models

## Technical Details
The DeepSeek parser is more robust than trying to find specific text patterns because:
- It works regardless of the exact phrasing DeepSeek uses
- It handles variations in formatting (e.g., `**Answer:**` vs `Answer:` vs just `(A)`)
- It uses the last occurrence to avoid matching examples in the prompt

## Testing
To test the fix, run:
```bash
.venv/bin/python run_nnsight_experiments.py --config configs/nnsight.yaml --no-cache
```

This will regenerate responses and use the new parser for DeepSeek models.