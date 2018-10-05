#!/bin/sh

cp _common.R.true _common.R
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

cp _common.R.false _common.R
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::pdf_book')"

rm _common.R
