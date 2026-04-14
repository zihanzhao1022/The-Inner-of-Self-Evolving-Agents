"""
LLM Internal Lens — Layer-wise Behavioral Dynamics Framework

Research focus: Map the relationship between output behaviors and internal
layer-wise representation dynamics across model scales.

Core modules:
    extractor    — Hook-based activation capture
    logit_lens   — Project hidden states to vocabulary space
    dynamics     — Layer-wise dynamics computation (bifurcation, intensity, geometry)
    behavior     — Behavior tagging, signature extraction, cross-behavior comparison
    cross_scale  — Cross-model-scale analysis, critical window hypothesis
    visualizer   — All visualization (dynamics, behaviors, cross-scale)
"""

try:
    from .extractor import ActivationExtractor, ExtractionResult
    from .logit_lens import LogitLens, LogitLensResult
    from .dynamics import LayerDynamics, DynamicsProfile
    from .behavior import BehaviorMapper, BehaviorReport
    from .cross_scale import CrossScaleAnalyzer, CriticalWindow
    from .compare import ReportComparator, ComparisonResult, CompareVisualizer
    from .report_io import save_report, load_report
    from .visualizer import LensVisualizer
except ImportError as e:
    import sys
    print(f"[llm_lens] Import failed: {e}", file=sys.stderr)
    print(f"  pip install torch transformers numpy matplotlib seaborn scikit-learn", file=sys.stderr)
    raise