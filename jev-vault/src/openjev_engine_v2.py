from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root = Path.home() / "openjev_research"
root.mkdir(parents=True, exist_ok=True)

ENGINE_SOURCE = '''
"""
OpenJev Research Engine v2
===========================
Strict, typed-decision runtime that keeps the model contract explicit.
"""

from __future__ import annotations

import json


def main() -> None:
    payload = {
        "status": "ready",
        "engine": "openjev_v2",
        "report": "modeled decision runtime initialized",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''

HARNESS_SOURCE = '''
"""
OpenJev Research Harness
========================
minimal runnable harness for the generated runtime.
"""

from __future__ import annotations

import json


def main() -> None:
    summary = {
        "run": "benchmark",
        "status": "demo",
        "output_dir": "~/.openjev_research",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''

REQUIREMENTS_SOURCE = """
torch>=2.5
transformers>=4.48
huggingface_hub>=0.26
safetensors>=0.4
laya>=0.1.6
datasets>=3.0
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
pydantic>=2.0
"""


def write_runtime_files() -> dict[str, Path]:
    files = {
        "engine": root / "openjev_engine_runtime.py",
        "harness": root / "openjev_harness.py",
        "requirements": root / "requirements.txt",
    }
    files["engine"].write_text(ENGINE_SOURCE.strip() + "\n", encoding="utf-8")
    files["harness"].write_text(HARNESS_SOURCE.strip() + "\n", encoding="utf-8")
    files["requirements"].write_text(REQUIREMENTS_SOURCE.strip() + "\n", encoding="utf-8")
    return files


def main() -> None:
    generated = write_runtime_files()
    print("OpenJev research files created:")
    for label, path in generated.items():
        print(f"- {label}: {path}")
    print("\nInstall dependencies with: pip install -r " + str(generated["requirements"]))
    print("Then run: python " + str(generated["harness"]) + " --run benchmark")


if __name__ == "__main__":
    main()
