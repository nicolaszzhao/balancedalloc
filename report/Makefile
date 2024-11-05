.PHONY: purge clean all

TARGET = nicolas.zhao-balancedalloc.tex

all: $(TARGET)

%.pdf: %.tex
	latexmk -shell-escape -pdf $<

clean:
	rm -f *.out *.fdb_latexmk *.fls *.log *.aux *.toc *.synctex.gz

purge: clean
	rm -f $(TARGET)
