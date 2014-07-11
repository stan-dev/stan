# The bib file
set(BIBFILES ../bibtex/all.bib)

# All inputed texfiles
set(INPUT_TEXFILES acknowledgements.tex advanced.tex appendices.tex
                   contributed.tex distributions.tex functions.tex
                   introduction.tex language.tex preface.tex programming.tex
                   references.tex stan-reference.tex title.tex)
set(IMAGE_DIRS R)

# Root tex file
set(MAIN_TEXFILE stan-reference.tex)

# Files which should be configured by cmake, e.g., to add the source dir path
set(CONFIG_TEXFILES stan-reference.tex acknowledgements.tex title.tex)

add_latex_document(${MAIN_TEXFILE}
                   BIBFILES ${BIBFILES}
                   INPUTS ${INPUT_TEXFILES}
                   IMAGE_DIRS ${IMAGE_DIRS}
                   CONFIGURE ${CONFIG_TEXFILES}
                   USE_INDEX DEFAULT_PDF)

