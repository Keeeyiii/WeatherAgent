"""Convert the markdown report into a self-contained, print-ready HTML file.

The HTML embeds every figure as base64, so it can be opened anywhere and
printed to PDF from any browser without external dependencies.
"""

from __future__ import annotations

import base64
import os
import re

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "研究报告.md")
TARGET = os.path.join(ROOT, "研究报告.html")

# LaTeX 公式在 HTML 中的等价写法（避免依赖外部 MathJax）
FORMULAS = {
    r"\text{error} = F - O,\qquad \text{bias} = \overline{F-O}":
        "误差 = F − O　　　偏差 bias = mean(F − O)",
    r"\text{MAE} = \overline{|F-O|},\qquad \text{RMSE} = \sqrt{\overline{(F-O)^2}}":
        "MAE = mean(|F − O|)　　　RMSE = sqrt(mean((F − O)²))",
    r"\widehat{\text{error}} = f(\text{预报气温},\ \text{湿度},\ \text{气压},\ \text{风速},\ \text{小时},\ \text{月份})":
        "预测误差 = f(预报气温, 湿度, 气压, 风速, 小时, 月份)",
    r"\text{订正后气温} = F - \widehat{\text{error}}":
        "订正后气温 = F − 预测误差",
    r"\text{误差} = -0.21 - 0.14 \times \text{观测距平}\qquad (r = -0.26,\ n = 23543)":
        "误差 = −0.21 − 0.14 × 观测距平　　(r = −0.26, n = 23543)",
}


def replace_math(text: str) -> str:
    def repl(match: re.Match) -> str:
        body = match.group(1).strip()
        rendered = FORMULAS.get(body, body)
        return f'<div class="formula">{rendered}</div>'

    return re.sub(r"\$\$(.+?)\$\$", repl, text, flags=re.S)


def embed_images(html: str) -> str:
    def repl(match: re.Match) -> str:
        src = match.group(2)
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            return match.group(0)
        with open(path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode()
        return f'{match.group(1)}data:image/png;base64,{encoded}"'

    return re.sub(r'(<img[^>]*src=")([^"]+)"', repl, html)


CSS = """
@page { size: A4; margin: 20mm 18mm; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans SC", sans-serif;
  color: #1b1b1b; line-height: 1.85; font-size: 11.2pt;
  max-width: 190mm; margin: 0 auto; padding: 12mm 8mm;
}
h1 { font-size: 20pt; line-height: 1.4; margin: 0 0 .4em 0; }
h2 { font-size: 15pt; margin: 1.9em 0 .7em 0; padding-bottom: .25em;
     border-bottom: 1.5px solid #123a5f; color: #123a5f; }
h3 { font-size: 12.6pt; margin: 1.5em 0 .5em 0; color: #123a5f; }
h4 { font-size: 11.6pt; margin: 1.2em 0 .4em 0; color: #2f7fb5; }
p { margin: .65em 0; text-align: justify; }
ul, ol { margin: .6em 0 .6em 1.4em; padding-left: .6em; }
li { margin: .3em 0; }
strong { color: #0d2b46; }
hr { border: none; border-top: 1px solid #d9d9d9; margin: 2em 0; }
table { border-collapse: collapse; width: 100%; margin: 1.1em 0 1.4em 0; font-size: 10.2pt; }
th { background: #123a5f; color: #fff; font-weight: 600; }
th, td { border: 1px solid #d9d9d9; padding: 7px 10px; vertical-align: middle; }
tbody tr:nth-child(even) { background: #f4f8fb; }
img { max-width: 100%; display: block; margin: 1.3em auto .4em auto; }
.formula {
  background: #f4f8fb; border-left: 3px solid #2f7fb5; padding: .7em 1em;
  margin: 1em 0; font-size: 11pt;
}
code { background: #f2f4f6; padding: .12em .35em; border-radius: 3px; font-size: 10pt; }
pre { background: #f7f8fa; border: 1px solid #e2e6ea; border-radius: 5px;
      padding: .9em 1.1em; overflow-x: auto; font-size: 9.6pt; line-height: 1.6; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #e07b39; background: #fff8ef;
             margin: 1em 0; padding: .7em 1.1em; }
h2, h3, h4 { page-break-after: avoid; }
table, img, pre, .formula { page-break-inside: avoid; }
"""


def main() -> None:
    with open(SOURCE, encoding="utf-8") as fh:
        text = fh.read()
    text = replace_math(text)
    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    html_body = embed_images(html_body)
    document = (
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n"
        "<meta charset=\"utf-8\" />\n"
        "<title>GFS 2 米气温预报误差的结构诊断与订正实验</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{html_body}\n</body>\n</html>\n"
    )
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(document)
    print("wrote", TARGET, f"({len(document) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
