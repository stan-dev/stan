FROM drmike/r-bookdown

RUN Rscript -e 'install.packages("tufte")'

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  r-cran-arm \
  r-cran-bayesplot


RUN apt-get update && apt-get install -y --no-install-recommends \
  latexmk \
  git \
  ssh \
  texlive-latex-recommended \
  texlive-latex-extra \
  lmodern

