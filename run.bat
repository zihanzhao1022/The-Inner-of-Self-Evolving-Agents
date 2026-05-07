@echo off
setlocal

REM ============================================================
REM  Layer-wise Behavioral Dynamics — full experiment matrix
REM    3 phase-1 runs (3B / 3B-Instruct / AZR-Coder-3B)
REM  + 3 pairwise comparisons (probe / intensity / bifurcation
REM                            / direction / BIC / norm
REM                            / per-head / class-centroid)
REM  All 6 outputs share one auto-generated timestamp.
REM
REM  Defaults:  --dataset condition_multiple  --max-per-class 200
REM  Override on the command line, e.g.
REM    run.bat --max-per-class 0          (full ~700/class)
REM    run.bat --dataset default          (legacy 32-prompt set)
REM    run.bat --skip-comparisons         (only phase 1)
REM    run.bat --skip-phase1 --timestamp 20260505-2228   (re-do compares)
REM ============================================================

python -m llm_lens.examples.run_all ^
    --dataset condition_multiple ^
    --max-per-class 200 ^
    %*

if errorlevel 1 (
    echo.
    echo [run.bat] run_all exited with errors.
)

pause
endlocal
