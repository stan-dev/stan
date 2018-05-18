#!/bin/sh

Rscript -e "bookdown::render_book('cover.Rmd', 'bookdown::gitbook')"
