PYTHON ?= python3

.PHONY: help install pipeline pipeline_no_experiment data preprocess topics classify aspects experiment visualize report compare clean

help:
	@echo "Available targets:"
	@echo "  make install                # Install dependencies"
	@echo "  make pipeline               # Full pipeline including extended binary experimentation"
	@echo "  make pipeline_no_experiment # Full pipeline without extended binary experimentation"
	@echo "  make data                   # Run data loading"
	@echo "  make preprocess             # Run text preprocessing"
	@echo "  make topics                 # Run LDA topic modeling"
	@echo "  make classify               # Run 5-class classifier"
	@echo "  make aspects                # Run aspect significance analysis"
	@echo "  make experiment             # Run extended binary experimentation (CV/AUC/ablation/errors)"
	@echo "  make compare                # Run binary model comparison + tiny tuning"
	@echo "  make visualize              # Generate figures"
	@echo "  make report                 # Run full main.py pipeline"
	@echo "  make clean                  # Remove generated output artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/data_loader.py

preprocess:
	$(PYTHON) src/preprocess.py

topics:
	$(PYTHON) src/topics.py

classify:
	$(PYTHON) src/classifier.py

aspects:
	$(PYTHON) src/aspects.py

experiment:
	$(PYTHON) src/experimentation_binary.py

compare:
	$(PYTHON) src/compare_models.py

visualize:
	$(PYTHON) src/visualize.py

pipeline: data preprocess topics classify aspects experiment visualize

pipeline_no_experiment: data preprocess topics classify aspects visualize

report:
	$(PYTHON) main.py

clean:
	rm -rf outputs/tables/*.csv outputs/figures/*.png outputs/pos_filter_evaluation.md outputs/report.md outputs/report_2page_refined.md
