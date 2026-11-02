"""Extract final test results without changing source logs or training code.

Run from the repository root: python notes/tools/summarize_pulse_logs.py
The workbook is authored separately from the generated JSON/CSV snapshot.
"""

import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
STAMP = "20261102"
MODELS = ("PULSE", "PULSE_SOFTS", "PULSE_SCF_SIMPLE", "PULSE_SCF")
FOLDERS = dict(zip(MODELS, ("Pulse", "PULSE_SOFTS", "PULSE_SCF_SIMPLE", "PULSE_SCF")))
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "weather",
            "Solar", "traffic", "PEMS03", "PEMS04", "PEMS07", "PEMS08")
METRICS = ("mse", "mae")


def parse_log(path, model):
    text = path.read_text(encoding="utf-8", errors="strict")
    namespaces = re.findall(r"^Namespace\(.*\)$", text, re.M)
    finals = re.findall(r"^mse:([^,\s]+),\s*mae:([^\s]+)", text, re.M)
    if len(namespaces) != 1 or len(finals) != 1 or "Traceback" in text:
        raise ValueError("Ambiguous or failed experiment: {}".format(path))
    node = ast.parse(namespaces[0], mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
        raise ValueError("Unexpected configuration: {}".format(path))
    args = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
    mse, mae = map(float, finals[0])
    if args["model"] != model or not all(math.isfinite(v) and v >= 0 for v in (mse, mae)):
        raise ValueError("Invalid model or metric: {}".format(path))
    if text.rfind("mse:") < text.rfind(">>>>>>>testing") or ">>>>>>>testing" not in text:
        raise ValueError("Missing final test phase: {}".format(path))
    lengths = re.search(r"_(\d+)_(\d+)\.log$", path.name)
    if not lengths or tuple(map(int, lengths.groups())) != (args["seq_len"], args["pred_len"]):
        raise ValueError("Filename and configuration differ: {}".format(path))
    splits = {}
    for name in ("train", "val", "test"):
        values = list(map(int, re.findall(r"^" + name + r"\s+(\d+)\s*$", text, re.M)))
        if not values or len(set(values)) != 1:
            raise ValueError("Missing or inconsistent sample counts: {}".format(path))
        splits[name] = values[0]
    epochs = re.findall(r"^Epoch:\s*(\d+), Steps:\s*(\d+)\s*\| Train Loss:\s*(\S+) Vali Loss:\s*(\S+) Test Loss:\s*(\S+)", text, re.M)
    durations = list(map(float, re.findall(r"^Epoch:\s*\d+ cost time:\s*(\S+)", text, re.M)))
    if not epochs or len(epochs) != len(durations):
        raise ValueError("Incomplete training history: {}".format(path))
    current_epoch, best_epoch = None, None
    for line in text.splitlines():
        match = re.match(r"Epoch:\s*(\d+), Steps:", line)
        if match:
            current_epoch = int(match[1])
        if "Saving model ..." in line:
            best_epoch = current_epoch
    if best_epoch is None:
        raise ValueError("No saved validation checkpoint: {}".format(path))
    return dict(model=model, dataset=path.parent.name, seq_len=args["seq_len"],
                pred_len=args["pred_len"], mse=mse, mae=mae, seed=args["random_seed"],
                epochs=len(epochs), best_epoch=best_epoch,
                early_stopped="Early stopping" in text,
                mean_epoch_seconds=mean(durations), total_epoch_seconds=sum(durations),
                splits=splits, args=args, source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_bytes=path.stat().st_size,
                final_metric_line=next(line for line in reversed(text.splitlines()) if line.startswith("mse:")))


def improvement(reference, result):
    return (reference - result) / reference * 100


def ranks(values):
    ordered = sorted(set(values))
    return [ordered.index(value) + 1 for value in values]


def ranked_cells(values, kind="md"):
    result = []
    for value, rank in zip(values, ranks(values)):
        text = "{:.6f}".format(value)
        if kind == "md":
            result.append("**{}**".format(text) if rank == 1 else "<u>{}</u>".format(text) if rank == 2 else text)
        else:
            result.append("<strong>{}</strong>".format(text) if rank == 1 else "<u>{}</u>".format(text) if rank == 2 else text)
    return result


def main():
    rows, excluded = [], []
    for model, folder in FOLDERS.items():
        for path in sorted((LOGS / folder).rglob("*.log")):
            if path.parent.name not in DATASETS:
                excluded.append(str(path.relative_to(ROOT)))
                continue
            rows.append(parse_log(path, model))
    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), r["pred_len"], MODELS.index(r["model"])))
    groups = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["seq_len"], row["pred_len"])
        if row["model"] in groups[key]:
            raise ValueError("Duplicate experiment: {}".format(key))
        groups[key][row["model"]] = row
    assert len(rows) == 192 and len(groups) == 48, (len(rows), len(groups))
    for dataset in DATASETS:
        expected = {12, 24, 48, 96} if dataset.startswith("PEMS") else {96, 192, 336, 720}
        for model in MODELS:
            assert {r["pred_len"] for r in rows if r["model"] == model and r["dataset"] == dataset} == expected
    comparison = []
    for key, group in groups.items():
        assert set(group) == set(MODELS), key
        baseline = group["PULSE"]
        for model, row in group.items():
            shared_keys = set(baseline["args"]) & set(row["args"]) - {"model", "model_id", "checkpoints"}
            differences = {k: [baseline["args"][k], row["args"][k]] for k in shared_keys
                           if baseline["args"][k] != row["args"][k]}
            assert not differences and row["splits"] == baseline["splits"], (key, model, differences)
            comparison.append(dict(dataset=key[0], seq_len=key[1], pred_len=key[2], model=model,
                                   differing_common_args=differences,
                                   args_only_in_baseline=sorted(set(baseline["args"]) - set(row["args"])),
                                   args_only_in_model=sorted(set(row["args"]) - set(baseline["args"]))))
            for metric in METRICS:
                row[metric + "_improvement_vs_pulse_pct"] = improvement(baseline[metric], row[metric])
                row[metric + "_rank"] = ranks([group[m][metric] for m in MODELS])[MODELS.index(model)]
    summaries = []
    for dataset in DATASETS:
        summaries.append(dict(dataset=dataset, **{
            model: {metric: mean(r[metric] for r in rows if r["dataset"] == dataset and r["model"] == model)
                    for metric in METRICS} for model in MODELS}))
    subsets = {"全部48组": list(groups.values()),
               "PEMS16组": [g for key, g in groups.items() if key[0].startswith("PEMS")],
               "其他32组": [g for key, g in groups.items() if not key[0].startswith("PEMS")]}
    statistics = []
    for label, subset in subsets.items():
        for model in MODELS:
            entry = dict(subset=label, n=len(subset), model=model)
            for metric in METRICS:
                entry[metric + "_wins_vs_base"] = sum(g[model][metric] < g["PULSE"][metric] for g in subset)
                entry[metric + "_firsts"] = sum(g[model][metric + "_rank"] == 1 for g in subset)
                entry[metric + "_seconds"] = sum(g[model][metric + "_rank"] == 2 for g in subset)
                entry[metric + "_mean_improvement_pct"] = mean(improvement(g["PULSE"][metric], g[model][metric]) for g in subset)
            entry["median_epoch_time_ratio_vs_base"] = median(g[model]["mean_epoch_seconds"] / g["PULSE"]["mean_epoch_seconds"] for g in subset)
            statistics.append(entry)
    flat_fields = ["dataset", "seq_len", "pred_len", "model", "mse", "mae", "mse_rank", "mae_rank",
                   "mse_improvement_vs_pulse_pct", "mae_improvement_vs_pulse_pct", "seed", "epochs",
                   "best_epoch", "early_stopped", "mean_epoch_seconds", "total_epoch_seconds", "source", "source_sha256"]
    csv_path = LOGS / ("四模型实验结果明细_" + STAMP + ".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    code_paths = [ROOT / "run.py", ROOT / "exp/exp_main.py"] + sorted((ROOT / "models").glob("PULSE*.py")) + sorted((ROOT / "layers/pulse_scf").rglob("*.py"))
    manifest = dict(report_date="2026-11-02", extracted_at=datetime.now().astimezone().isoformat(),
                    current_local_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    current_code_is_not_proof_of_training_revision=True,
                    current_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                    models=MODELS, datasets=DATASETS, rows=rows, dataset_means=summaries,
                    statistics=statistics, protocol_comparison=comparison, excluded_logs=excluded,
                    ignored_archives=["logs/logs.zip"],
                    ranking="Lower is better; rank raw values, preserve exact ties; display six decimals.",
                    relative_improvement="100*(PULSE-error - model-error)/PULSE-error; mean across paired configurations.")
    (LOGS / ("四模型实验结果溯源_" + STAMP + ".json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    title = "四模型实验结果汇总（2026-11-02）"
    lines = ["# " + title, "", "四个模型、十二个标准数据集、每个数据集四个预测长度，共 192 次完整训练与测试。PULSE 为基线，放在模型列首位。", "",
             "仅提取每份日志末尾测试阶段的 `mse:`、`mae:`，不使用训练中的 Test Loss 选优。MSE、MAE 均越小越好，分别排名；**最优加粗**，<u>次优加下划线</u>。按未四舍五入的原值排名，表中显示六位小数；精确并列同名次。", "",
             "当前每组只有随机种子 2024、itr=1。共同记录的参数和 train/val/test 样本数一致；尚未核验训练代码版本、数据文件哈希及模型专属有效配置。PEMS 的 12/24/48/96 与其他数据集的 96/192/336/720 分组分析。大桥数据的四份 PULSE 日志按既定范围排除，压缩包不重复统计。", "",
             "[Excel 表格](四模型实验结果汇总_20261102.xlsx) · [浏览器表格](四模型实验结果汇总_20261102.html) · [精确指标与日志路径](四模型实验结果明细_20261102.csv) · [配置和哈希溯源](四模型实验结果溯源_20261102.json)", "",
             "## 按数据集的四时距平均", "", "每个数据集内部对四个预测长度等权平均。不将不同数据集的原始误差直接合成一个总体 MSE。", "",
             "| 数据集 | 指标 | PULSE（Base） | PULSE_SOFTS | PULSE_SCF_SIMPLE | PULSE_SCF |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    html_tables = []
    def add_html_table(headers, records):
        html_tables.append("<table><thead><tr>" + "".join("<th>" + html.escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>" +
                           "".join("<tr>" + "".join("<td>" + c + "</td>" for c in cells) + "</tr>" for cells in records) + "</tbody></table>")
    mean_html = []
    for summary in summaries:
        for metric in METRICS:
            values = [summary[m][metric] for m in MODELS]
            lines.append("| " + " | ".join([summary["dataset"], metric.upper()] + ranked_cells(values)) + " |")
            mean_html.append([summary["dataset"], metric.upper()] + ranked_cells(values, "html"))
    add_html_table(["数据集", "指标", "PULSE (Base)", *MODELS[1:]], mean_html)
    lines += ["", "## 各预测长度完整结果", ""]
    detailed_html = []
    for dataset in DATASETS:
        lines += ["### " + dataset, "", "| L→H | 指标 | PULSE（Base） | PULSE_SOFTS | PULSE_SCF_SIMPLE | PULSE_SCF |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for (name, length, horizon), group in groups.items():
            if name != dataset:
                continue
            for metric in METRICS:
                values = [group[m][metric] for m in MODELS]
                lines.append("| " + " | ".join([str(length) + "→" + str(horizon), metric.upper()] + ranked_cells(values)) + " |")
                detailed_html.append([dataset, str(length) + "→" + str(horizon), metric.upper()] + ranked_cells(values, "html"))
        lines += ["", "来源：" + "，".join("[{}]({}/{}/)".format(m, FOLDERS[m], dataset) for m in MODELS), ""]
    add_html_table(["数据集", "L→H", "指标", "PULSE (Base)", *MODELS[1:]], detailed_html)
    lines += ["## 分组胜出与相对基线变化", "", "正数表示误差降低，负数表示误差升高。先对每个配对配置计算相对 PULSE 的变化，再在分组内等权平均。第一名次数和优于基线次数含义不同。", "",
              "| 分组 | 模型 | MSE 第一 / 配置数 | MAE 第一 / 配置数 | MSE 优于基线 | MAE 优于基线 | 平均 MSE 降低 | 平均 MAE 降低 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in statistics:
        lines.append("| {} | {} | {}/{} | {}/{} | {} | {} | {:+.2f}% | {:+.2f}% |".format(
            s["subset"], s["model"], s["mse_firsts"], s["n"], s["mae_firsts"], s["n"],
            "—" if s["model"] == "PULSE" else str(s["mse_wins_vs_base"]),
            "—" if s["model"] == "PULSE" else str(s["mae_wins_vs_base"]),
            s["mse_mean_improvement_pct"], s["mae_mean_improvement_pct"]))
    lines += ["", "## 结果分析与后续计划", "",
              "完整 PULSE_SCF 的收益主要集中在 PEMS：16 组中 MSE 最优 12 组、MAE 最优 15 组，相对 PULSE 的配对平均 MSE、MAE 分别降低 13.28%、9.57%。PEMS08 的 24/48 步仍有例外，不能写成全部 PEMS 配置均最优。", "",
              "其他 32 组中，PULSE 的 MSE、MAE 第一名次数分别为 16、21；PULSE_SCF 的平均 MSE、MAE 相对基线分别升高 4.64%、4.38%。ETTh1 四时距平均 MSE 升高 26.72%，Weather 则降低 5.80%。完整模型当前不具备全面优于基线的证据。", "",
              "PULSE_SCF_SIMPLE 相对 PULSE_SOFTS 改善 34/48 组 MSE，支持继续检验 FACT 的增量作用。完整 SCF 相比 SIMPLE 同时改变偏差输入和门控，现有结果无法分别证明两个机制。", "",
              "下一步先在预先固定的八组配置补跑两个种子，再做偏差输入与动态门控的 2×2 消融，补齐 PULSE＋FACT 对照；暂不继续叠加模块。详细证据、计算口径、失败案例与分阶段计划见[实验分析与下一步计划](../notes/四模型首轮实验分析与下一步计划_20261102.md)。", ""]
    (LOGS / ("四模型实验结果汇总_" + STAMP + ".md")).write_text("\n".join(lines), encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四模型实验结果汇总</title>
<style>body{font:15px/1.65 system-ui,'PingFang SC',sans-serif;color:#192438;background:#f7f8fb;margin:0}main{max-width:1260px;margin:auto;padding:36px}h1{font-size:26px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums;font-size:14px}th{background:#24364b;color:white;text-align:right;position:sticky;top:0;padding:10px}td{padding:7px 12px;text-align:right;border-bottom:1px solid #e4e8ee}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f5f7fa}strong{font-weight:800}u{text-decoration-thickness:1.5px;text-underline-offset:3px}a{color:#175c9c}.context{max-width:1100px}</style><main>
<h1>四模型实验结果汇总 · 2026-11-02</h1><p class="context">192 次完整测试，十二个标准数据集。PULSE 位于模型首列。MSE、MAE 分别排名，越低越好。<strong>最优加粗</strong>，<u>次优加下划线</u>，按原始精度排名，显示六位小数。每配置仅一个种子 2024。不同数据集不合成原始误差均值。</p>
<p><a href="四模型实验结果汇总_20261102.xlsx">下载 Excel</a>　<a href="四模型实验结果汇总_20261102.md">完整 Markdown 报告</a>　<a href="../notes/四模型首轮实验分析与下一步计划_20261102.md">分析与下一步计划</a></p>
<h2>每数据集四时距平均</h2>""" + html_tables[0] + "<h2>全部预测长度</h2>" + html_tables[1] + "</main></html>"
    (LOGS / ("四模型实验结果汇总_" + STAMP + ".html")).write_text(page, encoding="utf-8")
    print(json.dumps(dict(records=len(rows), groups=len(groups), per_model=dict(Counter(r["model"] for r in rows)),
                          protocol_mismatches=0, excluded=len(excluded)), ensure_ascii=False))


if __name__ == "__main__":
    main()
