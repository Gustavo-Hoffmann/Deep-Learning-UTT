#!/usr/bin/env python3
"""Extrai textos explicativos do transcript (uso único)."""
import json
import re
from pathlib import Path

TRANSCRIPT = Path(
    "/Users/Rodacki/.cursor/projects/Users-Rodacki-Desktop-Hoffmann-UTT/agent-transcripts/"
    "1a285f07-19fc-4bed-9b02-8452e1c84bfd/1a285f07-19fc-4bed-9b02-8452e1c84bfd.jsonl"
)
OUT = Path("/Users/Rodacki/Desktop/Hoffmann/UTT/_extracted_etapas.txt")

def strip_code_blocks(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "[diagrama omitido — ver scripts do pipeline]", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text

def main():
    found = {}
    with open(TRANSCRIPT) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("role") != "assistant":
                continue
            for part in obj.get("message", {}).get("content", []):
                if part.get("type") != "text":
                    continue
                text = part["text"]
                m = re.match(r"^# Etapa (\d+) — (.+?)(?:\n|$)", text)
                if m:
                    n = int(m.group(1))
                    if n not in found or len(text) > len(found[n]):
                        found[n] = strip_code_blocks(text)
                m2 = re.match(r"^## Etapa (\d+) — (.+?)(?:\n|$)", text)
                if m2:
                    n = int(m2.group(1))
                    if n not in found or len(text) > len(found[n]):
                        found[n] = strip_code_blocks(text.replace("## Etapa", "# Etapa", 1))

    with open(OUT, "w") as out:
        for n in sorted(found):
            out.write(f"\n\n{'='*70}\nETAPA {n}\n{'='*70}\n\n")
            out.write(found[n])
            out.write("\n")

    print(f"Extraídas etapas: {sorted(found.keys())}")
    for n in sorted(found):
        print(f"  {n}: {len(found[n])} chars")

if __name__ == "__main__":
    main()
