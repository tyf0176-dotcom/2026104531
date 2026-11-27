"""Rebuild the report tables and figures from results/*.jsonl.

    python tools/make_report.py

Writes results/report-tables.md plus report-latency-recall-dev.png and
report-test-selected.png so the numbers in the write-up always trace back to
the raw benchmark output.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "summary":
                continue
            row["_file"] = path.name
            rows.append(row)
    return rows


def label(row):
    build, search = row["build"], row["search"]
    if row["index"] == "exact":
        return "exact (baseline)"
    if row["index"] == "graph":
        return "HNSW M={}, efC={}".format(build.get("M"), build.get("efConstruction"))
    nbits = build.get("nbits", 0)
    if nbits == 0:
        return "IVF-Flat K={}".format(build.get("nlist"))
    return "IVF-PQ K={}, m={}, {}bit".format(build.get("nlist"), build.get("m"), nbits)


def knobs(row):
    build, search = row["build"], row["search"]
    if row["index"] == "exact":
        return "—"
    if row["index"] == "graph":
        return "efSearch={}".format(search.get("efSearch"))
    return "nprobe={}".format(search.get("nprobe"))


def table(rows, title):
    head = ("| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n")
    body = ""
    for r in sorted(rows, key=lambda r: (label(r), r["search_p50_ms"])):
        size = r.get("index_bytes", 0)
        if size == 0:
            space = "—"
        elif size < 1024 * 1024:
            space = "{:.0f} KB".format(size / 1024)
        else:
            space = "{:.1f} MB".format(size / 1024 / 1024)
        body += "| {} | {} | {:.0f} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n".format(
            label(r), knobs(r), r["build_ms"], space,
            r["ann_recall10"], r["file_hit5"], r["search_p50_ms"], r["search_p95_ms"])
    return "### {}\n\n".format(title) + head + body + "\n"


def curve(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    groups = {}
    for r in rows:
        groups.setdefault(label(r), []).append(r)
    style = {
        "HNSW": dict(marker="o", color="#1f4e79"),
        "IVF-Flat": dict(marker="s", color="#2e8b57"),
        "IVF-PQ": dict(marker="^", color="#c0504d"),
        "exact": dict(marker="*", color="#7f7f7f"),
    }
    for name, points in groups.items():
        key = next(k for k in style if name.startswith(k))
        points = sorted(points, key=lambda p: p["search_p50_ms"])
        ax.plot([p["search_p50_ms"] for p in points], [p["ann_recall10"] for p in points],
                label=name, markersize=5 if key != "exact" else 11,
                linewidth=1.2, **style[key])
    ax.set_xscale("log")
    ax.set_xlabel("ANN search latency, p50 (ms, log scale)")
    ax.set_ylabel("chunk Recall@10")
    ax.set_ylim(0.15, 1.03)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Dev split (10 queries): recall vs latency")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bars(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    names = [label(r) + "\n" + knobs(r) for r in rows]
    idx = range(len(rows))
    ax.bar([i - 0.2 for i in idx], [r["ann_recall10"] for r in rows], width=0.4, label="Recall@10", color="#1f4e79")
    ax.bar([i + 0.2 for i in idx], [r["file_hit5"] for r in rows], width=0.4, label="Hit@5", color="#c0504d")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, fontsize=6, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Test split (40 queries): selected configurations")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



def zoom(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    groups = {}
    for r in rows:
        groups.setdefault(label(r), []).append(r)
    colors = {8: "#1f4e79", 16: "#2e75b6", 32: "#9dc3e6"}
    for name, points in groups.items():
        points = sorted(points, key=lambda p: p["search_p50_ms"])
        m = points[0]["build"].get("M")
        ax.plot([p["search_p50_ms"] for p in points], [p["ann_recall10"] for p in points],
                marker="o", markersize=4, linewidth=1.1, label=name, color=colors.get(m, "#444"))
    ax.set_xscale("log")
    ax.set_xlabel("ANN search latency, p50 (ms, log scale)")
    ax.set_ylabel("chunk Recall@10")
    ax.set_title("Graph index, zoomed: the recall plateau")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



def load_online():
    rows = []
    for path in sorted(RESULTS.glob("online-*.jsonl")):
        tokens = []
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "query":
                tokens.append(row["tokens"])
            elif row.get("type") == "summary":
                row["tokens"] = tokens
                rows.append(row)
                tokens = []
    return rows


def online_table(rows):
    head = ("| 配置 | 查询参数 | 查询嵌入 (ms) | ANN 检索 (ms) | 生成 (ms) | 完整请求 (ms) | 框架/聚合 (ms) | 输出 token |\n"
            "|---|---|---:|---:|---:|---:|---:|---|\n")
    body = ""
    for r in rows:
        embed, search, llm, flow = r["embed_p50_ms"], r["search_p50_ms"], r["llm_p50_ms"], r["workflow_p50_ms"]
        body += "| {} | {} | {:.1f} | {:.2f} | {:.0f} | {:.0f} | {:.1f} | {} |\n".format(
            label(r), knobs(r) if r["index"] != "exact" else "—", embed, search, llm, flow,
            max(flow - embed - search - llm, 0), "/".join(str(t) for t in r["tokens"]))
    return head + body


def online_bars(rows, path):
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
    names = [label(r) + "\n" + (knobs(r) if r["index"] != "exact" else "full scan") for r in rows]
    embed = [r["embed_p50_ms"] for r in rows]
    search = [r["search_p50_ms"] for r in rows]
    llm = [r["llm_p50_ms"] for r in rows]
    other = [max(r["workflow_p50_ms"] - r["embed_p50_ms"] - r["search_p50_ms"] - r["llm_p50_ms"], 0) for r in rows]
    idx = range(len(rows))
    ax.bar(idx, embed, label="query embed", color="#2e75b6")
    ax.bar(idx, search, bottom=embed, label="ANN search", color="#2e8b57")
    base = [e + s for e, s in zip(embed, search)]
    ax.bar(idx, llm, bottom=base, label="LLM generate", color="#c0504d")
    base2 = [b + l for b, l in zip(base, llm)]
    ax.bar(idx, other, bottom=base2, label="framework", color="#bfbfbf")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, fontsize=6, rotation=18, ha="right")
    ax.set_ylabel("latency per request (ms, p50)")
    ax.set_title("Online RAG breakdown, 3 fixed test queries")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    rows = load()
    rows = [r for r in rows if r.get("mode") == "search"]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    text = "# 自动生成的实验数据表\n\n"
    text += "由 `tools/make_report.py` 从 `results/*.jsonl` 生成，请勿手工编辑。\n\n"
    for name, group in (("开发集（10 条查询）", dev), ("测试集（40 条查询）", test)):
        text += "## {}\n\n".format(name)
        for key, title in (("graph", "图索引 HNSW"), ("cluster", "聚类索引")):
            subset = [r for r in group if r["index"] == key]
            if subset:
                flat = [r for r in subset if r["build"].get("nbits", 0) == 0]
                pq = [r for r in subset if r["build"].get("nbits", 0) != 0]
                if flat:
                    text += table(flat, "{} · IVF-Flat".format(title) if key == "cluster" else title)
                if pq:
                    text += table(pq, "{} · IVF-PQ".format(title))
        exact = [r for r in group if r["index"] == "exact"]
        if exact:
            text += table(exact, "精确基线 (exact)")

    tables_text = text
    curve([r for r in dev if r["index"] != "exact"], RESULTS / "report-latency-recall-dev.png")
    graph_dev = [r for r in dev if r["index"] == "graph"]
    if graph_dev:
        zoom(graph_dev, RESULTS / "report-hnsw-zoom-dev.png")

    chosen = [r for r in test if (
        (r["index"] == "graph" and r["search"].get("efSearch") == 32)
        or (r["index"] == "cluster" and r["build"].get("nbits") == 0 and r["search"].get("nprobe") in (16, 64))
        or (r["index"] == "cluster" and r["build"].get("nbits") == 8 and r["search"].get("nprobe") == 64)
        or r["index"] == "exact")]
    if chosen:
        bars(chosen, RESULTS / "report-test-selected.png")

    online = load_online()
    if online:
        text += "## 在线 RAG 耗时（测试集 3 条固定查询）\n\n" + online_table(online) + "\n"
        online_bars(online, RESULTS / "report-online-breakdown.png")
    (RESULTS / "report-tables.md").write_text(text, encoding="utf8")

    # keep the table block inside report.md in sync with the raw results
    body = text.split("请勿手工编辑。", 1)[1].strip()
    report = ROOT / "report.md"
    if report.exists():
        current = report.read_text(encoding="utf8")
        start, end = "<!--TABLES:start-->", "<!--TABLES:end-->"
        if start in current and end in current:
            head, rest = current.split(start, 1)
            _, tail = rest.split(end, 1)
            report.write_text(head + start + "\n" + body + "\n" + end + tail, encoding="utf8")
    print("rows:", len(rows), "dev:", len(dev), "test:", len(test), "online:", len(online))


if __name__ == "__main__":
    main()
