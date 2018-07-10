BUILDING THE USER'S GUIDE

Prerequisites
1.  Install R packages:  bookdown
2.  Install on OS and ensure on PATH: pandoc, pandoc-citeproc,
pdflatex

Option 1:  Makefile
3.  In shell, change directories to the top-level <stan> directory
4.  make users-guide
5.  The final HTML and pdf will be renamed under <stan>/doc/users-guide

Option 2: Shell Script
3.  In a shell, change directories to this directory, <stan>/src/docs/manual
4.  In the shell, execute ./_build.sh
5.  The root of the final final HTML will be in _book/index.html and
the final pdf document will be in _book/_book.pdf.

