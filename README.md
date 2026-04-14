# LLM Internal Lens — Layer-wise Behavioral Dynamics Framework

Map the relationship between **output behaviors** and **internal layer-wise representation dynamics** across model scales.

## Research Questions

- **RQ1 (When):** At which layer does the model's internal representation diverge for different behaviors?
- **RQ2 (How):** Do different behavior types (harmful, safe, hallucination, sycophancy) have unique dynamics signatures?
- **RQ3 (Scale):** Is the critical decision window stable across model sizes? (Critical Window Hypothesis)

## Quick Start

```bash
pip install torch transformers numpy matplotlib seaborn scikit-learn scipy
```

### Phase 1: Single Model Analysis
```python
from llm_lens import ActivationExtractor, BehaviorMapper, LensVisualizer

extractor = ActivationExtractor("Qwen/Qwen2.5-3B")
mapper = BehaviorMapper(extractor)

mapper.add("How to pick a lock", "harmful")
mapper.add("What is photosynthesis", "safe")
# ... more samples ...

report = mapper.analyze()
report.print_summary()
# -> Best discriminative layer: 18 (depth ratio: 0.50, accuracy: 0.92)

viz = LensVisualizer()
viz.plot_probe_accuracy(report)
viz.plot_intensity_comparison(report)
viz.plot_behavior_trajectories(mapper)
```

### Phase 2: Cross-Scale Analysis
```python
from llm_lens import CrossScaleAnalyzer

analyzer = CrossScaleAnalyzer()
analyzer.add_report(report_3b, param_count=3.09)
analyzer.add_report(report_7b, param_count=7.62)

window = analyzer.test_critical_window()
window.print_summary()
# -> Critical window: [0.42, 0.58] (CV=0.08, STABLE)
```

### Run Full Experiment
```bash
python -m llm_lens.examples.run_experiment --phase 1 --model Qwen/Qwen2.5-3B
python -m llm_lens.examples.run_experiment --phase 2 --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B
```

## Architecture

```
llm_lens/
├── extractor.py      # Hook-based activation capture
├── logit_lens.py     # Vocabulary space projection
├── dynamics.py       # Layer-wise dynamics (intensity, bifurcation, trajectory)
├── behavior.py       # Behavior tagging → dynamics signatures → probe accuracy
├── cross_scale.py    # Critical Window Hypothesis testing
├── visualizer.py     # All plots (probe, intensity, trajectory, cross-scale)
└── examples/
    └── run_experiment.py   # Phase 1 + Phase 2 experiment script
```

## Module Pipeline

```
Prompts + Tags          →  extractor.py  →  dynamics.py
("harmful"/"safe"/...)     (hook capture)    (per-layer metrics)
                                                  ↓
                                            behavior.py
                                            (aggregate by tag,
                                             probe accuracy,
                                             bifurcation timing)
                                                  ↓
                                            cross_scale.py
                                            (compare across 3B/7B/14B,
                                             test critical window)
                                                  ↓
                                            visualizer.py
                                            (all figures)
```

## Key Outputs

- **Probe accuracy curve**: which layer best separates behaviors
- **Processing intensity profiles**: how much each layer transforms input, by behavior
- **Bifurcation analysis**: where harmful/safe representations diverge
- **PCA trajectories**: geometric shape of internal processing paths
- **Critical window**: stable depth ratio range across model scales
- **JSON reports**: machine-readable results for downstream analysis
