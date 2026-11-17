import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

def require(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")

if __name__ == "__main__":
    
    # Check a README.md exists
    readme = Path("README.md")
    require(readme.is_file(), "Missing README.md")
    print("PASS: README.md exists")


    # Check the submitted report.
    report = Path("report/report.pdf")
    require(report.is_file(), "Missing report/report.pdf")

    try:
        reader = PdfReader(report)
        require(len(reader.pages) > 0, "report.pdf has no pages")
    except Exception as exc:
        sys.exit(f"FAIL: Cannot read report.pdf: {exc}")
    
    print("PASS: report/report.pdf exists and is readable")

    # Check the results path 
    require(results.is_file(), "Missing results/results.json")

    try:
        with open(results) as f:
            data = json.load(f)
    except Exception as exc:
        sys.exit(f"FAIL: Cannot read results/results.json: {exc}")

    print("PASS: results/results.json exists and is in correct format")

