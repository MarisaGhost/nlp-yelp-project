"""
Main script to run the full Yelp NLP project pipeline.

This script runs all steps in order:
1) Data loading
2) Text preprocessing
3) Topic modeling (LDA)
4) Rating prediction (classification)
5) Aspect-based analysis
6) Extended binary experimentation (CV/AUC/ablation/error analysis)
7) Visualization

All outputs will be saved to the outputs/ folder.
"""

import subprocess
import sys

print("Starting Yelp NLP project pipeline...\n")

steps = [
    ("Data loading", [sys.executable, "src/data_loader.py"]),
    ("Text preprocessing", [sys.executable, "src/preprocess.py"]),
    ("Topic modeling (LDA)", [sys.executable, "src/topics.py"]),
    ("Rating prediction (classification)", [sys.executable, "src/classifier.py"]),
    ("Aspect-based analysis", [sys.executable, "src/aspects.py"]),
    (
        "Extended binary experimentation (CV/AUC/ablation/error analysis)",
        [sys.executable, "src/experimentation_binary.py"],
    ),
    ("Visualization", [sys.executable, "src/visualize.py"]),
]

for name, command in steps:
    print(f"Running step: {name}")
    print(f"Command: {' '.join(command)}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nError occurred during step: {name}")
        print("Pipeline stopped.")
        sys.exit(1)

    print(f"Finished step: {name}\n")

print("Pipeline completed successfully.\n")
print("Results saved in:")
print("- outputs/tables/")
print("- outputs/figures/")
print("- outputs/report.md")
