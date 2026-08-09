# Code Reorganization Plan for Post-Hoc Reasoning Repository

## Overview
This document outlines a comprehensive plan to reorganize the codebase to achieve:
- Better separation between backend implementations (TransformerLens vs NNsight)
- Reduced code duplication
- More maintainable and extensible architecture
- Clearer project structure

## Current Structure Issues

### Problems Identified
1. **Asymmetric Organization**: NNsight code is well-organized in `nnsight_utils/` while TransformerLens code is scattered
2. **Code Duplication**: Both backends have separate experiment runners with ~80% identical code
3. **Mixed Concerns**: Backend-specific code mixed with shared algorithms
4. **Flat Structure**: TransformerLens implementation lacks proper modularization

## Target Structure

```
src/
├── backends/                      # Backend-specific implementations
│   ├── __init__.py               # Backend factory/registry
│   ├── base.py                   # Abstract base classes
│   ├── transformer_lens/
│   │   ├── __init__.py
│   │   ├── models.py             # ChatModel wrapper
│   │   ├── utils.py              # Activation extraction, generation
│   │   └── steering.py           # TL-specific steering
│   └── nnsight/
│       ├── __init__.py
│       ├── models.py             # NNsightChatModel wrapper
│       ├── utils.py              # Basic utilities
│       ├── steering.py           # Steering implementation
│       └── unified/              # Current nnsight_utils content
│           ├── core/
│           ├── probes/
│           ├── steering/
│           └── ...
│
├── methods/                      # Backend-agnostic algorithms
│   ├── __init__.py
│   ├── probes/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract probe interface
│   │   ├── logistic.py          # Logistic regression probes
│   │   └── contrastive.py       # CAA probes
│   └── steering/
│       ├── __init__.py
│       ├── base.py              # Abstract steering interface
│       ├── caa_single_layer.py  # CAA single-layer method
│       ├── caa_incremental.py   # CAA layer-incremental
│       └── logistic_steering.py # Logistic regression steering
│
├── core/                         # Shared core functionality
│   ├── __init__.py
│   ├── cache_manager.py         # Experiment caching
│   ├── config.py                # Configuration management
│   ├── data_loading.py          # Dataset loading
│   ├── parsing_utils.py         # Response parsing
│   ├── memory_utils.py          # Memory management
│   └── visualizer.py            # Visualization
│
├── integrations/                 # External service integrations
│   ├── __init__.py
│   └── wandb.py                 # Weights & Biases integration
│
└── runners/                      # Experiment runners
    ├── __init__.py
    ├── base.py                  # Abstract experiment runner
    └── unified.py               # Single unified runner
```

## Detailed Migration Plan

### Phase 1: Create Backend Directory Structure
**Goal**: Establish the new directory structure and move backend-specific files

#### Steps:
1. Create directory structure:
   ```
   src/backends/
   src/backends/transformer_lens/
   src/backends/nnsight/
   ```

2. Move TransformerLens files:
   - `src/models.py` → `src/backends/transformer_lens/models.py`
   - `src/utils.py` → `src/backends/transformer_lens/utils.py`
   - Extract TL-specific steering from `utils.py` → `src/backends/transformer_lens/steering.py`

3. Move NNsight files:
   - `src/nnsight_models.py` → `src/backends/nnsight/models.py`
   - `src/nnsight_steering.py` → `src/backends/nnsight/steering.py`
   - `src/nnsight_utils.py` → `src/backends/nnsight/utils.py`
   - `src/nnsight_utils/` → `src/backends/nnsight/unified/`

4. Create base classes:
   - `src/backends/base.py` with abstract interfaces for models and steering

5. Update imports in moved files

**Test After Phase 1**:
```bash
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml
```

### Phase 2: Extract Shared Methods
**Goal**: Move algorithmic code to shared methods directory

#### Steps:
1. Create directory structure:
   ```
   src/methods/
   src/methods/probes/
   src/methods/steering/
   ```

2. Move steering methods:
   - `src/steering_methods.py` → `src/methods/steering/methods.py`
   - Split into individual files:
     - `CAASingleLayerSteering` → `src/methods/steering/caa_single_layer.py`
     - `CAALayerIncrementalSteering` → `src/methods/steering/caa_incremental.py`
     - `LogisticRegressionSteering` → `src/methods/steering/logistic_steering.py`

3. Extract probe training logic:
   - From `experiment_runner.py` and `nnsight_experiment_runner.py`
   - Create `src/methods/probes/logistic.py`
   - Create `src/methods/probes/contrastive.py`

4. Create abstract base classes:
   - `src/methods/probes/base.py`
   - `src/methods/steering/base.py`

5. Update imports

**Test After Phase 2**:
```bash
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml
```

### Phase 3: Reorganize Core Utilities
**Goal**: Move shared utilities to core directory

#### Steps:
1. Create directory structure:
   ```
   src/core/
   src/integrations/
   ```

2. Move core utilities:
   - `src/cache_manager.py` → `src/core/cache_manager.py`
   - `src/config.py` → `src/core/config.py`
   - `src/data_loading.py` → `src/core/data_loading.py`
   - `src/parsing_utils.py` → `src/core/parsing_utils.py`
   - `src/memory_utils.py` → `src/core/memory_utils.py`
   - `src/visualizer.py` → `src/core/visualizer.py`

3. Move integrations:
   - `src/wandb_integration.py` → `src/integrations/wandb.py`

4. Update all imports across the codebase

**Test After Phase 3**:
```bash
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml
```

### Phase 4: Unify Experiment Runners
**Goal**: Create a single unified experiment runner

#### Steps:
1. Create directory structure:
   ```
   src/runners/
   ```

2. Extract common logic from both runners:
   - Create `src/runners/base.py` with abstract base class
   - Create `src/runners/unified.py` with single implementation

3. Use dependency injection for backend-specific operations:
   - Model creation via backend factory
   - Activation extraction via backend interface
   - Steering application via backend interface

4. Update experiment runner imports in:
   - `run_transformer_lens_experiments.py`
   - `run_nnsight_experiments.py`
   - Both should use the same unified runner with different backend configs

**Test After Phase 4**:
```bash
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml
```

### Phase 5: Update Entry Points and Clean Up
**Goal**: Update main entry points and remove old files

#### Steps:
1. Update `run_transformer_lens_experiments.py`:
   - Import from new locations
   - Use unified runner with TransformerLens backend

2. Update `run_nnsight_experiments.py`:
   - Import from new locations
   - Use unified runner with NNsight backend

3. Update `src/main.py` if used

4. Remove old files:
   - `src/experiment_runner.py` (replaced by unified)
   - `src/nnsight_experiment_runner.py` (replaced by unified)
   - Clean up any duplicate code

5. Update model_factory.py to use new backend structure

**Final Test**:
```bash
# Run full test suite
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml

# Run with original configs to ensure compatibility
python run_transformer_lens_experiments.py --config configs/transformer_lens.yaml
python run_nnsight_experiments.py --config configs/nnsight.yaml
```

## Import Update Examples

### Before:
```python
# In experiment_runner.py
from models import ChatModel
from utils import generate_with_steering
from steering_methods import create_steering_method

# In nnsight_experiment_runner.py
from nnsight_models import NNsightChatModel
from nnsight_steering import generate_with_nnsight_steering
```

### After:
```python
# In unified runner
from backends.transformer_lens.models import ChatModel
from backends.nnsight.models import NNsightChatModel
from methods.steering import create_steering_method
from core.cache_manager import ExperimentCache
from core.config import ExperimentRunConfig
```

## Testing Strategy

### Test Configs
We have two minimal test configs:
- `configs/test_transformer_lens.yaml` - Tests TransformerLens backend
- `configs/test_nnsight.yaml` - Tests NNsight backend

Both use:
- Model: google/gemma-2-2b-it
- Dataset: anachronisms (10 train/10 test)
- Alpha range: [0, 2]
- Max gen: 2
- Cache: disabled

### Test Commands After Each Phase
```bash
# Quick smoke test
python run_transformer_lens_experiments.py --config configs/test_transformer_lens.yaml
python run_nnsight_experiments.py --config configs/test_nnsight.yaml

# If those pass, test with original configs
python run_transformer_lens_experiments.py --config configs/transformer_lens.yaml --datasets anachronisms --train-size 10 --test-size 10
python run_nnsight_experiments.py --config configs/nnsight.yaml --datasets anachronisms --train-size 10 --test-size 10
```

## Benefits of Reorganization

1. **Better Maintainability**: Clear separation of concerns
2. **Reduced Duplication**: Single implementation of shared algorithms
3. **Easier Testing**: Can test algorithms independently of backends
4. **Extensibility**: Easy to add new backends (JAX, pure PyTorch, etc.)
5. **Cleaner Dependencies**: Backend-specific imports isolated
6. **Better Documentation**: Structure reflects architecture

## Rollback Plan

If issues arise during reorganization:
1. Git provides natural rollback via commits
2. Test after each phase to catch issues early
3. Keep original files until new structure is verified
4. Can run old and new in parallel during transition

## Success Criteria

The reorganization is successful when:
1. Both test configs run without errors
2. Results match original implementation
3. Code duplication is significantly reduced
4. New structure is intuitive and well-organized
5. Adding a new backend would be straightforward