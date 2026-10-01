from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRECTION = {"long": 1, "positive": 1, "neutral": 0, "short": -1, "negative": -1}

TYPE_MAP = {
    "News": "other",
    "Earnings": "earnings",
    "Guidance": "guidance",
    "ManagementChange": "management",
    "ProductLaunch": "production",
    "Production": "production",
}

def infer_severity(title: str, direction: str) -> int:
    t = title.lower()
    if direction in {"short", "negative"}:
        if any(k in t for k in ["investigation", "default", "fraud", "suspend", "halt", "warning", "loss", "cut"]):
            return 3
        return 2
    if any(k in t for k in ["repurchase", "acquire", "profit growth", "permit", "resource estimate", "contract"]):
        return 2
    return 1

def main():
    p=argparse.ArgumentParser(description="Normalize Longbridge security_facts JSON to event CSV")
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--symbol",required=True)
    p.add_argument("--industry",default="")
    args=p.parse_args()

    raw=json.loads(Path(args.input).read_text(encoding="utf-8"))
    facts=raw.get("facts",raw if isinstance(raw,list) else [])
    lines=["event_time,symbol,direction,severity,event_type,industry,source_quality,title"]
    for f in facts:
        title=((f.get("nl_info") or {}).get("title") or "").replace("
"," ").replace(","," ")
        direction=f.get("direction","neutral")
        event_type=TYPE_MAP.get(f.get("fact_type","News"),"other")
        severity=infer_severity(title,direction)
        lines.append(",".join([
            f.get("occur_time",""),
            args.symbol,
            str(DIRECTION.get(direction,0)),
            str(severity),
            event_type,
            args.industry,
            "1.0",
            title,
        ]))
    Path(args.output).write_text("
".join(lines)+"
",encoding="utf-8")

if __name__=="__main__":
    main()
