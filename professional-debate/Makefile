FILE=ondrej-pesek-professional-debate-2020

all:
	pdflatex $(FILE)
	bibtex $(FILE)

clean:
	rm -f *~ *.aux *.log *.out *.bbl *.blg *.loa *.lof \
	*.toc *.lot *.lol

distclean:
	make clean
	rm -f $(FILE).pdf
