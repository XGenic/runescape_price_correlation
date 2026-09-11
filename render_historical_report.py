"""Render saved historical notebook outputs, without fetching or fitting anything."""
import argparse
from html import escape
from pathlib import Path
import re

import nbformat
from nbconvert.filters import markdown2html


CSS = """
:root{--bg:#f5f4ef;--ink:#18323b;--muted:#53666c;--teal:#17675f;--line:#d9dfd9;--white:#fff}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:25px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,'Segoe UI',sans-serif}a{color:var(--teal);text-underline-offset:4px}a:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid var(--teal);outline-offset:4px}.wrap{width:min(1220px,calc(100% - 72px));margin:auto}.masthead{border-bottom:1px solid var(--line);padding:20px 0}.masthead .wrap{display:flex;justify-content:space-between;gap:20px;align-items:center}.brand{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:750}.actions{display:flex;gap:20px;font-size:12px;align-items:center}button{font:inherit;border:1px solid var(--line);border-radius:5px;padding:8px 12px;background:transparent;color:var(--ink);cursor:pointer}.skip{position:absolute;top:-70px;background:white;padding:12px;z-index:10}.skip:focus{top:10px}.hero{padding:60px 0 42px}.eyebrow{font-size:11px;text-transform:uppercase;font-weight:750;color:var(--teal);letter-spacing:.14em}.hero h1{font:normal clamp(42px,5.5vw,72px)/1.08 Georgia,serif;letter-spacing:-.035em;margin:18px 0 24px}.hero h1 span{color:var(--teal)}.dek{font-size:19px;max-width:820px;color:var(--muted)}.verdict{padding:24px 28px;background:#213e49;color:#edf2f0;border-radius:12px;margin-top:28px}.verdict strong{display:block;text-transform:uppercase;font-size:11px;letter-spacing:.11em;color:#c6dfd6;margin-bottom:8px}.verdict p{margin:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:22px}.stat{padding:22px;background:white}.stat small{display:block;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}.stat b{display:block;font-size:35px;letter-spacing:-.03em;line-height:1.4;margin:6px 0}.stat span{font-size:12px;color:var(--muted)}.layout{display:grid;grid-template-columns:205px minmax(0,1fr);gap:38px;padding-bottom:50px}.toc{position:sticky;top:25px;align-self:start;padding-top:24px;font-size:12px}.toc a{display:block;text-decoration:none;color:var(--muted);padding:8px 0}.toc a:hover{color:var(--teal)}.toc-label{font-size:10px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px;font-weight:700}.note{margin-top:22px;border-top:1px solid var(--line);padding-top:15px;font-size:11px;color:var(--muted)}section.research{border-top:1px solid var(--line);padding:34px 0;scroll-margin-top:24px}h2{font:normal 31px/1.25 Georgia,serif;letter-spacing:-.02em;margin:0 0 22px}h3{font-size:21px;line-height:1.35;margin:27px 0 15px}p{margin:0 0 17px}strong{font-weight:650}li{margin:8px 0}code{background:#e8eee8;padding:2px 4px;border-radius:3px;font-size:.86em;overflow-wrap:anywhere}pre{background:white;border:1px solid var(--line);border-radius:8px;overflow-x:auto;padding:17px;font-size:12px;line-height:1.7}.table-scroll{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:9px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.6}th,td{text-align:left!important;border:0;border-bottom:1px solid #e8ece7;padding:11px 13px;vertical-align:top}thead th{background:#edf1ed;font-size:11px;color:var(--muted)}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.observation{border-left:3px solid var(--teal);padding:16px 20px;background:#e8f0e9;margin:22px 0;font-size:14px}.observation p:last-child{margin-bottom:0}.observation h3{margin-top:0}figure{margin:25px 0;border:1px solid var(--line);border-radius:10px;background:white;overflow:hidden}figure img{display:block;max-width:100%;width:100%;height:auto;padding:10px}figcaption{border-top:1px solid var(--line);padding:12px 18px;font-size:12px;color:var(--muted)}details{border:1px solid var(--line);border-radius:8px;margin:20px 0;background:white;overflow:hidden}summary{padding:14px 18px;font-size:13px;cursor:pointer;font-weight:600}details .table-scroll{margin:0;border-radius:0;border-inline:0;border-bottom:0}.warning{background:#fcf0df;border-left:3px solid #a66b31;padding:15px 18px;font-size:14px;margin:20px 0}.provenance{font-size:12px;color:var(--muted);overflow-wrap:anywhere}footer{border-top:1px solid var(--line);padding:22px 0;font-size:12px;color:var(--muted)}.anchor-link{display:none}
@media(max-width:900px){.wrap{width:calc(100% - 40px)}.layout{grid-template-columns:150px minmax(0,1fr);gap:24px}.stat{padding:17px}.stat b{font-size:29px}}
@media(max-width:680px){.wrap{width:calc(100% - 32px)}.masthead .wrap{gap:12px}.brand{font-size:10px}.actions{gap:10px;font-size:10px}.hero{padding:35px 0 28px}.hero h1{font-size:46px}.dek{font-size:17px}.verdict{padding:22px;font-size:14px}.stats{grid-template-columns:1fr 1fr}.stat b{font-size:30px}.layout{display:block}.toc{display:flex;position:static;overflow-x:auto;white-space:nowrap;gap:20px;padding:12px 0 22px}.toc-label,.toc .note{display:none}.toc a{font-size:11px}section.research{padding:27px 0}h2{font-size:28px}figure img{padding:4px}th,td{padding:10px 12px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{@page{size:A4;margin:15mm}body{background:white;color:#111;font-size:10pt}.wrap{width:100%}.actions,.toc,.skip{display:none}.hero{padding:20px 0}.hero h1{font-size:34pt}.dek{font-size:12pt}.layout{display:block}.verdict{background:#edf1ed;color:#111;border:1px solid #ccc}.verdict strong{color:#111}.stat{padding:12px}.stat b{font-size:24pt}section.research{padding:20px 0}h2{font-size:23pt;break-after:avoid}h3{break-after:avoid}figure{break-inside:avoid}figure img{max-height:215mm;object-fit:contain}figcaption{font-size:9pt}.table-scroll{overflow:visible}table{font-size:8pt;overflow-wrap:anywhere}th,td{padding:6px 8px}.observation,.warning,.stats{break-inside:avoid}details::details-content{display:block;content-visibility:visible}details>*{display:block}p{orphans:3;widows:3}}
"""


def _joined(value):
    return "".join(value) if isinstance(value, list) else str(value)


def render_report(notebook_path, output_path):
    """Bind headline metrics to metadata embedded in the same executed outputs."""
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    metadata = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise ValueError("Execute all historical notebook cells before rendering a report.")
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                raise ValueError("Notebook contains an execution error; refusing a success report.")
            if "historical_report" in out.get("metadata", {}):
                metadata = out.metadata.historical_report
    if metadata is None:
        raise ValueError("No completed historical report metadata. Run the historical evaluation and summary first.")
    sections, body, title = [], [], "Historical study"
    figure_number = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            text = cell.source
            heading = re.search(r"^## (.+)$", text, re.MULTILINE)
            if heading:
                if body:
                    sections.append((title, "\n".join(body)))
                    body = []
                title = heading.group(1)
            text = re.sub(r"^# [^\n]+\n?", "", text, flags=re.MULTILINE)
            body.append(markdown2html(text))
        elif cell.cell_type == "code":
            for out in cell.get("outputs", []):
                data = out.get("data", {})
                if "image/png" in data:
                    figure_number += 1
                    label = f"Figure {figure_number} · {title}"
                    encoded = _joined(data["image/png"]).replace("\n", "")
                    body.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(label, quote=True)}">'
                                f'<figcaption>{escape(label)}. Embedded from the saved historical notebook output.</figcaption></figure>')
                elif "text/markdown" in data:
                    body.append('<div class="observation">' + markdown2html(_joined(data["text/markdown"])) + '</div>')
                elif "text/html" in data:
                    html = _joined(data["text/html"])
                    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
                    if "<table" in html:
                        # Keep details toggles outside the scrollable table region.
                        html = re.sub(r"(<table\b[^>]*>.*?</table>)",
                                      r'<div class="table-scroll" role="region" tabindex="0" aria-label="Research result table">\1</div>',
                                      html, flags=re.DOTALL)
                    body.append(html)
                elif "text/plain" in data:
                    body.append('<pre>' + escape(_joined(data['text/plain'])) + '</pre>')
                elif out.output_type == "stream" and out.get("text", "").strip():
                    body.append('<pre>' + escape(out.text) + '</pre>')
    if body:
        sections.append((title, "\n".join(body)))
    navigation = ''.join(f'<a href="#section-{i}">{escape(name)}</a>' for i, (name, _) in enumerate(sections))
    content = ''.join(f'<section class="research" id="section-{i}">{html}</section>' for i, (_, html) in enumerate(sections))
    audit = metadata['audit']
    iid = metadata['shuffle_sensitivity'][0]['pvalue']
    accuracy = metadata['evaluation_accuracy']
    baseline = metadata['evaluation_baseline_accuracy']
    score = 'Unavailable' if accuracy is None else f'{accuracy:.2%}'
    baseline_text = 'No eligible final outcomes' if baseline is None else f'{baseline:.2%} matched baseline'
    notebook_link = escape(Path(notebook_path).name, quote=True)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An audited historical RuneScape bond guide-price study, with dependence-aware lag sensitivity and a separate transaction-feed comparison.">
<title>The longer record — RuneScape bonds and the S&amp;P 500</title><style>{CSS}</style></head>
<body><a class="skip" href="#study">Skip to the study</a><header class="masthead"><div class="wrap"><div class="brand">Alternative data / Historical research</div><div class="actions"><a href="rs_bond_sp500_study.html">Original study</a><a href="{notebook_link}">Notebook</a><button type="button" onclick="window.print()">Print / PDF</button></div></div></header>
<main class="wrap" id="study"><div class="hero"><div class="eyebrow">OSRS guide prices · {escape(audit['postlaunch_start'])} — {escape(audit['postlaunch_end'])}</div><h1>A longer record.<br><span>A tougher test.</span></h1>
<p class="dek">The historical archive exists. This separate study audits it, tests bond returns across {audit['equity_sessions']:,} equity sessions, and asks whether guide-price observations agree with the transaction-average feed.</p>
<div class="verdict"><strong>The evidence, not the headline correlation</strong><p>{escape(metadata['conclusion'])}</p></div>
<div class="stats"><div class="stat"><small>Post-launch guide days</small><b>{audit['postlaunch_daily_quotes']:,}</b><span>{audit['missing_calendar_days']} missing calendar dates audited</span></div><div class="stat"><small>Best training lag</small><b>{metadata['best_lag']:+d} sessions</b><span>Pearson r = {metadata['best_corr']:.4f}</span></div><div class="stat"><small>Whole-search IID p</small><b>{iid:.3f}</b><span>5- and 20-session block checks below</span></div><div class="stat"><small>Historical evaluation</small><b>{score}</b><span>{baseline_text}; n = {metadata['evaluation_n']}</span></div></div>
<div class="warning"><strong>Not a pristine holdout:</strong> {metadata['exposed_sessions']} of {metadata['evaluation_calendar_sessions']} final-period session dates overlap the already-exposed original study. Chronological fitting and a frozen ledger prevent within-run lookahead, not prior researcher exposure.</div></div>
<div class="layout"><nav class="toc" aria-label="Historical report sections"><div class="toc-label">In this report</div>{navigation}<p class="note">Saved notebook results only. No API requests, fitting or evaluation run when this page is opened.</p></nav><article>{content}</article></div>
<p class="provenance"><strong>Snapshot fingerprint:</strong> <code>{escape(metadata['dataset_fingerprint'])}</code><br>All figures, tables, observations and headline metrics come from the same executed notebook. The report works offline; companion notebook and original-study links require those local files.</p></main>
<footer><div class="wrap">Historical GE-price experiment · Original transaction-price study preserved · Exploratory research, not investment advice</div></footer></body></html>'''
    Path(output_path).write_text(html, encoding="utf-8")
    return {"output": str(output_path), "sections": len(sections), "figures": figure_number,
            "fingerprint": metadata['dataset_fingerprint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="rs_bond_sp500_historical_investigation.ipynb")
    parser.add_argument("--output", default="rs_bond_sp500_historical_study.html")
    args = parser.parse_args()
    print(render_report(args.notebook, args.output))


if __name__ == "__main__":
    main()
