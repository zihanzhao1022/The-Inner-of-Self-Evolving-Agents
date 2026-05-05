#!/usr/bin/env python3
"""
Example: Analyze Qwen2.5-3B internal representations.

Usage:
    python -m llm_lens.examples.qwen_example

This script demonstrates all four analysis views:
1. Single-model internal change rate
2. Logit Lens token prediction evolution
3. Two-model divergence (base vs instruct)
4. PCA trajectory visualization
"""

import os
import sys
import torch

# Add parent to path if running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_lens import ActivationExtractor, LogitLens, LayerMetrics, LensVisualizer


def example_single_model():
    """Analyze a single model's internal processing."""
    print("=" * 60)
    print("Example 1: Single Model Analysis (Qwen2.5-3B)")
    print("=" * 60)

    # ── Step 1: Load model and extract activations ──
    extractor = ActivationExtractor(
        "Qwen/Qwen2.5-3B",
        dtype=torch.float32,   # fp32 for reproducibility
        capture=("residual",),
    )

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")
    result = extractor.run(prompt)
    print(f"Tokens: {result.tokens}")
    print(f"Shape: {result.num_layers} layers × {result.seq_len} tokens × {result.hidden_dim}d")

    # ── Step 2: Logit Lens ──
    print("\n--- Logit Lens ---")
    lens = LogitLens(extractor)
    lens_result = lens.decode_all_layers(result, top_k=5)
    print(lens_result.summary_table())

    # ── Step 3: Internal change rate ──
    print("\n--- Layer-wise Change Rate ---")
    change = LayerMetrics.compute_change_profile(result, token_idx=-1)
    most_active = int(change.inter_layer_l2.argmax())
    print(f"Most active transition: Layer {most_active} → {most_active + 1}")
    print(f"  L2 distance: {change.inter_layer_l2[most_active]:.4f}")
    print(f"  Cosine sim:  {change.inter_layer_cosine[most_active]:.4f}")

    # ── Step 4: Visualize ──
    viz = LensVisualizer()

    fig1 = viz.plot_logit_lens_heatmap(lens_result, save_path="output_logit_lens.png")
    print("\nSaved: output_logit_lens.png")

    fig2 = viz.plot_change_profile(change, save_path="output_change_rate.png")
    print("Saved: output_change_rate.png")

    fig3 = viz.plot_trajectory(result, save_path="output_trajectory_single.png")
    print("Saved: output_trajectory_single.png")

    return extractor, result


def example_compare_models():
    """Compare base vs instruct model on the same prompt."""
    print("\n" + "=" * 60)
    print("Example 2: Base vs Instruct Comparison")
    print("=" * 60)

    model_a_name = "Qwen/Qwen2.5-3B"            # base
    model_b_name = "Qwen/Qwen2.5-3B-Instruct"   # instruct / chat

    prompt = "How to pick a lock"

    # ── Extract from both models ──
    ext_a = ActivationExtractor(model_a_name, capture=("residual",))
    result_a = ext_a.run(prompt)

    ext_b = ActivationExtractor(model_b_name, capture=("residual",))
    result_b = ext_b.run(prompt)

    # ── Compute divergence ──
    div = LayerMetrics.compute_divergence(result_a, result_b, per_token=True)
    print(f"\nBifurcation layer: {div.bifurcation_layer}")
    print(f"Cosine sim range: {div.cosine_sim.min():.4f} ~ {div.cosine_sim.max():.4f}")

    # ── Logit Lens comparison ──
    lens_a = LogitLens(ext_a)
    lens_b = LogitLens(ext_b)
    lr_a = lens_a.decode_all_layers(result_a)
    lr_b = lens_b.decode_all_layers(result_b)

    print("\n--- Model A (Base) Logit Lens ---")
    print(lr_a.summary_table())
    print("\n--- Model B (Instruct) Logit Lens ---")
    print(lr_b.summary_table())

    # ── Visualize ──
    viz = LensVisualizer()

    viz.plot_divergence_curve(div, save_path="output_divergence.png")
    print("\nSaved: output_divergence.png")

    viz.plot_logit_lens_heatmap(lr_a, lens_result_b=lr_b, save_path="output_logit_compare.png")
    print("Saved: output_logit_compare.png")

    viz.plot_trajectory(result_a, result_b, save_path="output_trajectory_compare.png")
    print("Saved: output_trajectory_compare.png")

    viz.plot_per_token_divergence(div, save_path="output_per_token_div.png")
    print("Saved: output_per_token_div.png")


def example_multi_prompt():
    """Analyze multiple prompts to find patterns."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Prompt Pattern Analysis")
    print("=" * 60)

    extractor = ActivationExtractor("Qwen/Qwen2.5-3B", capture=("residual",))
    lens = LogitLens(extractor)

    prompts = [
        "The meaning of life is",
        "2 + 2 =",
        "Once upon a time, there was a",
        "import torch\nmodel =",
    ]

    for p in prompts:
        print(f"\n--- Prompt: '{p}' ---")
        result = extractor.run(p)
        lr = lens.decode_all_layers(result, top_k=3)
        # Show key transition points
        for i in range(lr.num_layers):
            tok, prob = lr.top_tokens[i][0]
            if i == 0 or tok != lr.top_tokens[i - 1][0][0]:
                print(f"  Layer {i:2d}: {repr(tok):>15s}  (p={prob:.3f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Internal Lens Examples")
    parser.add_argument(
        "--mode",
        choices=["single", "compare", "multi", "all"],
        default="single",
        help="Which example to run",
    )
    args = parser.parse_args()

    if args.mode in ("single", "all"):
        example_single_model()
    if args.mode in ("compare", "all"):
        example_compare_models()
    if args.mode in ("multi", "all"):
        example_multi_prompt()

    print("\nDone!")
