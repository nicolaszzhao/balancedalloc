.PHONY: purge clean all generate-plots

TARGET = nicolas.zhao-balancedalloc.pdf

all: $(TARGET)


%.pdf: %.tex generate-plots
	latexmk -shell-escape -pdf $<


generate-plots:
	python3 ../src/driver.py

clean:
	rm -f *.out *.fdb_latexmk *.fls *.log *.aux *.toc *.synctex.gz

purge: clean
	rm -f $(TARGET)
