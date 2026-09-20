.PHONY: help preprocess optimize train evaluate figures paper clean

help:
	@echo "Targets:"
	@echo "  preprocess   Run uniform preprocessing on all three datasets"
	@echo "  optimize     Run Optuna TPE (50 trials per dataset)"
	@echo "  train        Train all seven models on all three datasets"
	@echo "  evaluate     Compute metrics, statistical tests, Pareto frontier"
	@echo "  figures      Regenerate all paper figures"
	@echo "  paper        Compile LaTeX source to PDF"
	@echo "  clean        Remove generated artefacts"

preprocess:
	python -m src.preprocessing --dataset all

optimize:
	python -m src.optimize --dataset all --trials 50 --seed 42

train:
	python -m src.train --dataset all --models all

evaluate:
	python -m src.evaluate --dataset all
	python -m src.statistics
	python -m src.pareto

figures:
	python figures/source/fig2_convergence.py
	python figures/source/fig3_macro_f1_bars.py
	python figures/source/fig4_radar.py
	python figures/source/fig5_pareto.py
	python figures/source/fig6_confusion.py
	python figures/source/fig7_roc.py

paper:
	cd paper && pdflatex cose.tex && bibtex cose && pdflatex cose.tex && pdflatex cose.tex

clean:
	rm -rf data/processed/* results/metrics/* results/significance/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
