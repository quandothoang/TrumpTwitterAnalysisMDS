.PHONY: all clean

# Top level target
all: report/count_report.html

# Final report
report/count_report.html: report/count_report.qmd \
  results/figure/isles.png \
  results/figure/abyss.png \
  results/figure/last.png \
  results/figure/sierra.png
	quarto render report/count_report.qmd

# Figures
results/figure/isles.png: results/isles.dat scripts/plotcount.py
	python scripts/plotcount.py --input_file=$< --output_file=$@

results/figure/abyss.png: results/abyss.dat scripts/plotcount.py
	python scripts/plotcount.py --input_file=$< --output_file=$@

results/figure/last.png: results/last.dat scripts/plotcount.py
	python scripts/plotcount.py --input_file=$< --output_file=$@

results/figure/sierra.png: results/sierra.dat scripts/plotcount.py
	python scripts/plotcount.py --input_file=$< --output_file=$@

# Word count data files
results/isles.dat: data/isles.txt scripts/wordcount.py
	python scripts/wordcount.py --input_file=$< --output_file=$@

results/abyss.dat: data/abyss.txt scripts/wordcount.py
	python scripts/wordcount.py --input_file=$< --output_file=$@

results/last.dat: data/last.txt scripts/wordcount.py
	python scripts/wordcount.py --input_file=$< --output_file=$@

results/sierra.dat: data/sierra.txt scripts/wordcount.py
	python scripts/wordcount.py --input_file=$< --output_file=$@

# Clean everything created by the pipeline
clean:
	rm -f results/*.dat
	rm -f results/figure/*.png
	rm -f report/count_report.html