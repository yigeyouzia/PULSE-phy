#!/usr/bin/env python3
"""Read local forecasting sources and record parameter/configuration evidence.

No training, checkpoint writes, shell-script execution or source modifications.
"""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from models import PULSE, PULSE_FACT


def parser_defaults(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            try:
                name = ast.literal_eval(node.args[0]).lstrip("-")
                keyword = next(k for k in node.keywords if k.arg == "default")
                result[name] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, StopIteration):
                pass
    return result


def shell_commands(path, defaults):
    """Parse the simple dataset scripts as text, without evaluating shell code."""
    env, result = {}, []
    for line in path.read_text().replace("\\\n", " ").splitlines():
        match = re.match(r"^([A-Za-z_]\w*)=(.*)$", line.strip())
        if match:
            env[match[1]] = match[2].strip("\"'")
            continue
        if not line.strip().startswith("python -u run.py"):
            continue
        tokens = shlex.split(line)
        values = dict(defaults)
        for i, token in enumerate(tokens[:-1]):
            if not token.startswith("--"):
                continue
            raw = re.sub(r"\$\{?(\w+)\}?", lambda m: env.get(m[1], m[0]), tokens[i + 1])
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            values[token[2:]] = value
        result.append(values)
    return result


def counts(model):
    result = {name: sum(p.numel() for p in child.parameters())
              for name, child in model.named_children()}
    result.update({name: p.numel() for name, p in model.named_parameters(recurse=False)})
    total = sum(p.numel() for p in model.parameters())
    assert sum(result.values()) == total
    return {"total_trainable": total, "components": result}


def fused(config):
    with redirect_stdout(io.StringIO()):
        return PULSE_FACT.Model(SimpleNamespace(**config))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2024)
    np.random.seed(2024)
    defaults = parser_defaults(ROOT / "run.py")
    fact_defaults = parser_defaults(WORKSPACE / "FACT-main/run.py")
    rows, sources = [], [ROOT / "models/PULSE.py", ROOT / "models/PULSE_FACT.py",
        ROOT / "run.py", ROOT / "exp/exp_main.py", ROOT / "layers/pulse_fact/fact_block.py",
        ROOT / "layers/pulse_fact/vendor/fact_conv.py", WORKSPACE / "FACT-main/models/FACT.py",
        WORKSPACE / "FACT-main/run.py", WORKSPACE / "FACT-main/FACT.sh"]
    for script in sorted((ROOT / "scripts/PULSE_FACT").glob("*.sh")):
        sources.append(script)
        base_script = ROOT / "scripts/PULSE" / script.name
        sources.append(base_script)
        base_commands = shell_commands(base_script, defaults)
        for index, config in enumerate(shell_commands(script, defaults)):
            shared_fields = ("seq_len", "pred_len", "enc_in", "d_model", "inv_len", "patch_size",
                "dsa", "dsb", "ksize", "time_dim", "batch_size", "learning_rate",
                "train_epochs", "patience", "rec_lambda", "auxi_lambda", "random_seed")
            assert all(config[k] == base_commands[index][k] for k in shared_fields)
            base = PULSE.Model(SimpleNamespace(**config))
            model = fused(config)
            base_count, model_count = counts(base), counts(model)
            assert model_count["total_trainable"] - base_count["total_trainable"] == 278272
            rows.append({"script": str(script.relative_to(ROOT)), "config": config,
                "pulse": base_count, "pulse_fact": model_count,
                "grid_side": model.fact_block.block.channel_patch,
                "fact_settings": model.get_extra_state()["settings"]})
            del base, model

    spec = importlib.util.spec_from_file_location("fact_readonly_audit", WORKSPACE / "FACT-main/models/FACT.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    reference_config = dict(fact_defaults, task_name="long_term_forecast", seq_len=96, pred_len=96, enc_in=7)
    reference = source.Model(SimpleNamespace(**reference_config))
    reference_counts = counts(reference)
    del reference

    tiny = dict(defaults, seq_len=24, pred_len=12, enc_in=7, d_model=8, inv_len=24,
        patch_size=6, time_dim=1, dsa=1, dsb=1, ksize=3, label_len=0,
        use_amp=False, use_multi_gpu=False)
    torch.use_deterministic_algorithms(True)
    model = fused(tiny).train()
    batch = (torch.randn(2, 24, 7), torch.arange(2), torch.randn(2, 24, 1), torch.randn(2, 12, 1))
    output, mixing = model(*batch)
    target = torch.randn_like(output)
    lam, permutation = mixing
    target = lam * target + (1 - lam) * target[permutation]
    loss = (torch.fft.fft(output, dim=1) - torch.fft.fft(target, dim=1)).abs().mean()
    loss.backward()
    assert output.shape == (2, 12, 7) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        assert model(*batch).shape == (2, 12, 7)

    small_fact = dict(fact_defaults, task_name="long_term_forecast", seq_len=24, pred_len=12,
        enc_in=7, d_model=16, d_ff=32, dilation=[1, 2], num_kernels=2, core=0.5, freq="n")
    reference = source.Model(SimpleNamespace(**small_fact)).eval()
    with torch.no_grad():
        assert reference(torch.randn(2, 24, 7), None, None, None).shape == (2, 12, 7)
    source_zero_branch = "not checked"
    try:
        reference = source.Model(SimpleNamespace(**dict(small_fact, core=0.0))).eval()
        with torch.no_grad():
            reference(torch.randn(2, 24, 7), None, None, None)
    except AttributeError as error:
        source_zero_branch = type(error).__name__ + ": " + str(error)

    output_path = ROOT / "notes/data/pulse_fact_parameter_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"date": "2026-11-04",
        "audit_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "source_sha256": {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "pulse_cli_defaults": defaults, "fact_cli_defaults": fact_defaults,
        "fact_default_reference": {"config": reference_config, **reference_counts},
        "dataset_profiles": rows,
        "smoke_check": {"pulse_fact_train_eval_shape": [2, 12, 7], "sam_fft_backward": "passed_cpu",
            "strict_cpu_determinism": True, "fact_mixed_branch_shape": [2, 12, 7],
            "fact_core_zero_result": source_zero_branch, "training_run_started": False}},
        ensure_ascii=False, indent=2) + "\n")
    print("Saved", output_path)
    print("Profile count:", len(rows), "; FACT delta: 278272 for every profile")
    print("FACT default reference parameters:", reference_counts["total_trainable"])
    print("CPU shape/SAM FFT backward checks passed; FACT core=0:", source_zero_branch)


if __name__ == "__main__":
    main()
