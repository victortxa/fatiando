# Build, package and clean Fatiando
PY := python
PIP := pip
NOSE := nosetests

help:
	@echo "Commands:"
	@echo ""
	@echo "    build         build the extension modules inplace"
	@echo "    cython        generate C code from Cython files before building"
	@echo "    docs          build the html documentation"
	@echo "    view-docs     show the html docs on Firefox"
	@echo "    test          run the test suite (including doctests)"
	@echo "    test-par      run tests in parallel with all available cores"
	@echo "    pep8          check for PEP8 style compliance"
	@echo "    pep8-stats    print a summary of the PEP8 check"
	@echo "    coverage      calculate test coverage using Coverage"
	@echo "    package       create source distributions"
	@echo "    upload        upload source distribuitions to PyPI"
	@echo "    clean         clean up build and generated files"
	@echo "    clean-docs    clean up the docs build"
	@echo ""

.PHONY: build
build:
	$(PY) setup.py build_ext --inplace

cython:
	$(PY) setup.py build_ext --inplace --cython

docs: clean
	pip install --upgrade --quiet sphinx sphinx-rtd-theme || \
		pip install --user --upgrade --quiet sphinx sphinx-rtd-theme
	cd doc; make html

view-docs:
	firefox doc/_build/html/index.html &

.PHONY: test
test: build
	pip install --upgrade --quiet nose pep8 || \
		pip install --user --upgrade --quiet nose pep8
	$(NOSE) --with-doctest -v fatiando/ test/

test-par: build
	pip install --upgrade --quiet nose pep8 || \
		pip install --user --upgrade --quiet nose pep8
	$(NOSE) --with-doctest -v --processes=`nproc` fatiando/ test/

coverage: build
	pip install --upgrade --quiet nose pep8 coverage || \
		pip install --user --upgrade --quiet nose pep8 coverage
	$(NOSE) --with-doctest --with-coverage --cover-package=fatiando fatiando/ \
		test/

pep8:
	pep8 fatiando test cookbook

pep8-stats:
	pep8 --statistics -qq fatiando test cookbook

package: test-par
	$(PY) setup.py sdist --formats=zip,gztar

upload:
	python setup.py register sdist --formats=zip,gztar upload

clean:
	find . -name "*.so" -exec rm -v {} \;
	#find "fatiando" -name "*.c" -exec rm -v {} \;
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST README.txt CITATION.txt
	# Trash generated by the doctests
	rm -rvf mydata.txt mylogfile.log
	# The stuff fetched by the cookbook recipes
	rm -rvf logo.png cookbook/logo.png
	rm -rvf crust2.tar.gz cookbook/crust2.tar.gz
	rm -rvf bouguer_alps_egm08.grd cookbook/bouguer_alps_egm08.grd

clean-docs:
	cd doc; make clean
