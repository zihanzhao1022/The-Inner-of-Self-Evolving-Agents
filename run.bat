@echo off
python -m llm_lens.examples.run_experiment --phase 1 --model Qwen/Qwen2.5-3B --output results/Qwen2.5-3B/20260422-1053
python -m llm_lens.examples.run_experiment --phase 1 --model Qwen/Qwen2.5-3B-Instruct --output results/Qwen2.5-3B-Instruct/20260422-1053
python -m llm_lens.examples.run_experiment --phase 1 --model andrewzh/Absolute_Zero_Reasoner-Coder-3b --output results/AZR-Coder-3B/20260422-1053

python -m llm_lens.examples.run_comparison --base results/Qwen2.5-3B/20260422-1053/report_Qwen_Qwen2.5-3B.json --altered results/Qwen2.5-3B-Instruct/20260422-1053/report_Qwen_Qwen2.5-3B-Instruct.json --output results/Qwen2.5-3B_vs_Qwen2.5-3B-Instruct_20260422-1053

python -m llm_lens.examples.run_comparison --base results/Qwen2.5-3B/20260422-1053/report_Qwen_Qwen2.5-3B.json --altered results/AZR-Coder-3B/20260422-1053/report_andrewzh_Absolute_Zero_Reasoner-Coder-3b.json --output results/Qwen2.5-3B_vs_AZR-Coder-3B_20260422-1053

python -m llm_lens.examples.run_comparison --base results/Qwen2.5-3B-Instruct/20260422-1053/report_Qwen_Qwen2.5-3B-Instruct.json --altered results/AZR-Coder-3B/20260422-1053/report_andrewzh_Absolute_Zero_Reasoner-Coder-3b.json --output results/Qwen2.5-3B-Instruct_vs_AZR-Coder-3B_20260422-1053
pause
