#!/usr/bin/env python3
"""
Enhanced W&B table logging for better tracking of steering results.

This shows how to update the log_steering_results_table method to include
more detailed tracking of parseability and answer changes.
"""

def enhanced_log_steering_results_table():
    """Enhanced version of log_steering_results_table with better tracking."""
    
    print("ENHANCED TABLE FORMAT FOR W&B")
    print("="*60)
    
    code = '''
def log_steering_results_table(self, 
                             results: List[Dict[str, Any]], 
                             alpha: float,
                             direction: str):
    """Log a table of steering results for easy viewing."""
    if self.disabled or not results:
        return
        
    # Create enhanced table data with more detailed tracking
    table_data = []
    for r in results[:30]:  # Show up to 30 rows for better overview
        original = r.get("original_answer", "N/A")
        steered = r.get("new_answer", r.get("steered_answer", "N/A"))
        target = r.get("target_answer", "N/A")
        category = r.get("category", "unknown")
        
        # Determine specific outcomes
        parsable = steered in ["yes", "no"]  # Is the answer parseable?
        answer_changed = original != steered  # Did the answer change?
        success = (steered == target)  # Did we achieve the target?
        
        # Create status emoji for quick visual scanning
        if not parsable:
            status_emoji = "❌"  # Unparsed
        elif success:
            status_emoji = "✅"  # Success
        elif answer_changed:
            status_emoji = "🔄"  # Changed but wrong direction
        else:
            status_emoji = "➖"  # No change
        
        # Add row with detailed information
        table_data.append([
            alpha,
            direction,
            original,
            steered,
            target,
            parsable,
            answer_changed,
            success,
            category,
            status_emoji
        ])
    
    # Create table with comprehensive columns
    table = wandb.Table(
        columns=[
            "Alpha", 
            "Direction", 
            "Original", 
            "Steered", 
            "Target", 
            "Parsable",
            "Changed",
            "Success",
            "Category",
            "Status"
        ],
        data=table_data
    )
    
    # Log with descriptive key
    wandb.log({f"steering/results_table/alpha_{alpha}_{direction}": table})

# Alternative: Summary statistics table
def log_steering_summary_table(self, all_results: Dict[float, Dict[str, List]]):
    """Create a summary table across all alpha values."""
    if self.disabled:
        return
    
    summary_data = []
    for alpha, directions in all_results.items():
        for direction, results in directions.items():
            if not results:
                continue
                
            total = len(results)
            parsable = sum(1 for r in results if r.get("steered_answer") in ["yes", "no"])
            changed = sum(1 for r in results if r.get("original_answer") != r.get("steered_answer"))
            success = sum(1 for r in results if r.get("category") == "success")
            
            summary_data.append([
                alpha,
                direction,
                total,
                parsable,
                f"{parsable/total*100:.1f}%",
                changed,
                f"{changed/total*100:.1f}%",
                success,
                f"{success/total*100:.1f}%"
            ])
    
    summary_table = wandb.Table(
        columns=[
            "Alpha",
            "Direction", 
            "Total",
            "Parsable",
            "Parse%",
            "Changed",
            "Change%",
            "Success",
            "Success%"
        ],
        data=summary_data
    )
    
    wandb.log({"steering/summary_table": summary_table})
'''
    
    print(code)

def show_example_output():
    """Show what the table would look like in W&B."""
    print("\n" + "="*60)
    print("EXAMPLE TABLE OUTPUT IN W&B")
    print("="*60)
    
    print("""
Example of what you'll see in the W&B interface:

┌───────┬───────────┬──────────┬─────────┬────────┬──────────┬─────────┬─────────┬──────────┬────────┐
│ Alpha │ Direction │ Original │ Steered │ Target │ Parsable │ Changed │ Success │ Category │ Status │
├───────┼───────────┼──────────┼─────────┼────────┼──────────┼─────────┼─────────┼──────────┼────────┤
│  -5.0 │ yes_to_no │   yes    │   no    │   no   │   True   │  True   │  True   │ success  │   ✅   │
│  -5.0 │ yes_to_no │   yes    │   no    │   no   │   True   │  True   │  True   │ success  │   ✅   │
│  -5.0 │ yes_to_no │   yes    │   yes   │   no   │   True   │  False  │  False  │ failure  │   ➖   │
│  -5.0 │ yes_to_no │   yes    │ maybe   │   no   │   False  │  True   │  False  │ unparsed │   ❌   │
│  -5.0 │ yes_to_no │   yes    │   no    │   no   │   True   │  True   │  True   │ success  │   ✅   │
│   5.0 │ no_to_yes │   no     │   yes   │  yes   │   True   │  True   │  True   │ success  │   ✅   │
│   5.0 │ no_to_yes │   no     │   no    │  yes   │   True   │  False  │  False  │ failure  │   ➖   │
│   5.0 │ no_to_yes │   no     │ [garb.] │  yes   │   False  │  True   │  False  │ unparsed │   ❌   │
└───────┴───────────┴──────────┴─────────┴────────┴──────────┴─────────┴─────────┴──────────┴────────┘

Summary Table:
┌───────┬───────────┬───────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Alpha │ Direction │ Total │ Parsable │ Parse% │ Changed │ Change% │ Success │ Success% │
├───────┼───────────┼───────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│  -10  │ yes_to_no │  100  │    85    │ 85.0%  │   75    │  75.0%  │   70    │  70.0%   │
│  -5   │ yes_to_no │  100  │    90    │ 90.0%  │   65    │  65.0%  │   60    │  60.0%   │
│  -2   │ yes_to_no │  100  │    92    │ 92.0%  │   45    │  45.0%  │   40    │  40.0%   │
│   2   │ no_to_yes │  100  │    92    │ 92.0%  │   45    │  45.0%  │   40    │  40.0%   │
│   5   │ no_to_yes │  100  │    90    │ 90.0%  │   65    │  65.0%  │   60    │  60.0%   │
│  10   │ no_to_yes │  100  │    85    │ 85.0%  │   75    │  75.0%  │   70    │  70.0%   │
└───────┴───────────┴───────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘

Key insights from the enhanced table:
1. **Parsable**: Shows if the model produced a valid yes/no answer
2. **Changed**: Shows if steering had ANY effect on the answer
3. **Success**: Shows if steering achieved the TARGET answer
4. **Status Emoji**: Quick visual indicator of outcome

This helps you identify:
- When steering changes answers but in the wrong direction (🔄)
- When steering has no effect at all (➖)
- When the model produces unparseable outputs (❌)
- When steering works perfectly (✅)
""")

def show_filtering_examples():
    """Show how to use W&B's table filtering."""
    print("\n" + "="*60)
    print("W&B TABLE FILTERING EXAMPLES")
    print("="*60)
    
    print("""
In the W&B interface, you can filter and sort these tables:

1. **Find all unparsed examples**:
   - Click on the Parsable column
   - Filter: Parsable == False
   - See all examples that produced garbled output

2. **Find cases where answer changed but failed**:
   - Filter: Changed == True AND Success == False
   - These are cases where steering had an effect but wrong direction

3. **Sort by success rate**:
   - Use the summary table
   - Sort by Success% column
   - Identify which alpha values work best

4. **Export for analysis**:
   - Click the download button on any table
   - Export as CSV for further analysis
   - Share specific filtered views with colleagues

5. **Create custom views**:
   - Save filtered table views
   - Name them (e.g., "High Alpha Failures")
   - Share with your team

Example W&B query for the table:
```
table["Parsable"] == False AND table["Alpha"] > 5
```
This finds all unparsed responses at high alpha values.
""")

def show_integration_update():
    """Show how to update the existing code."""
    print("\n" + "="*60)
    print("HOW TO UPDATE YOUR INTEGRATION")
    print("="*60)
    
    print("""
To add this enhanced table tracking:

1. Replace the log_steering_results_table method in wandb_integration.py
   with the enhanced version shown above.

2. In experiment_runner.py, make sure you're passing complete result dicts
   that include all the necessary fields:
   
   ```python
   result = {
       "original_answer": example["pred_answer"],
       "steered_answer": parsed_answer,  # or "new_answer" 
       "target_answer": target_answer,
       "category": category,  # "success", "failure", or "unparsed"
       "original_response": example["response"],
       "steered_response": generation
   }
   ```

3. Call the table logging after each steering condition:
   
   ```python
   # After processing all examples for an alpha/direction
   if self.wandb_logger:
       self.wandb_logger.log_steering_results_table(
           results=results_yes,
           alpha=alpha_yes,
           direction="yes"
       )
   ```

4. Optionally, create a summary table at the end:
   
   ```python
   # After all steering experiments
   if self.wandb_logger:
       all_results = {}  # Collect from cache or variables
       self.wandb_logger.log_steering_summary_table(all_results)
   ```

This gives you comprehensive tracking of:
- Whether responses are parseable (valid yes/no)
- Whether steering changed the answer at all
- Whether the change was in the desired direction
- Quick visual status indicators
""")

if __name__ == "__main__":
    enhanced_log_steering_results_table()
    show_example_output()
    show_filtering_examples()
    show_integration_update()
    
    print("\n" + "="*60)
    print("BENEFITS OF ENHANCED TABLES")
    print("="*60)
    print("""
With these enhanced tables, you can:

1. **Debug steering failures**: Quickly identify if failures are due to:
   - Unparseable outputs (model breaking down)
   - No effect (steering too weak)
   - Wrong direction (steering vector issues)

2. **Track effectiveness**: See at a glance:
   - Parse rate: Is the model still coherent?
   - Change rate: Is steering having any effect?
   - Success rate: Is it working as intended?

3. **Identify patterns**: 
   - Do certain alpha values cause more parsing failures?
   - Are some examples resistant to steering?
   - Is there a sweet spot for alpha values?

4. **Share insights**: 
   - Export filtered views for specific phenomena
   - Create reports highlighting interesting patterns
   - Collaborate on debugging specific issues
""")