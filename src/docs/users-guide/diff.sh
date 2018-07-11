#!/bin/bash -e

OLD_VERSION=$1
if [ -z "$OLD_VERSION" ]; then
	echo 'You must specify a tag or SHA or something that git can interpret as a starting point'
	exit 1
fi
NEW_VERSION=$2
if [ -z "$NEW_VERSION" ]; then
	echo 'You must specify a tag or SHA or something that git can interpret as an endpoint (such as HEAD)'
	exit 2
fi

ADDED_LINE=^\{+.*+}$
IGNORE_COMMENT_LINES='/%/d'
DELETE_BRACEPLUS='s/^{+//g'
DELETE_PLUSBRACE='s/+}$//g'
IGNORE_LINES_WITH_LEADING_SLASH='/^\\.*$/d'
IGNORE_LINES_WITH_LEADING_GREATER_THAN='/^>.*$/d'
IGNORE_LINES_WITH_LEADING_LEFT_BRACE='/^{/d'
IGNORE_LINES_WITH_LEADING_RIGHT_BRACE='/^}/d'
IGNORE_LINES_WITH_RIGHT_BRACKET='/^.*\[.*/d'
IGNORE_LINES_THAT_START_WITH_DATA='/^data\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_TRANSFORMED_DATA='/^transformed\sdata\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_PARAMETERS='/^parameters\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_TRANSFORMED_PARAMETERS='/^transformed\sparameters\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_MODEL='/^model\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_GENERATED_QUANTITIES='/^generated\squantities\s*{\s*$/d'
IGNORE_LINES_THAT_START_WITH_CODE_COMMENTS='/^[#/].*$/d'

TEX_FILES=`ls *.tex`
#TEX_FILES=commands.tex
UNIQUE_LINES=0
for TEX_FILE in $TEX_FILES; do
	if [ "stan-reference.tex" = "$TEX_FILE" ]; then
		continue
	fi
	git diff --unified=0 --inter-hunk-context=0 --word-diff --color=never $OLD_VERSION $NEW_VERSION $TEX_FILE | \
	grep ${ADDED_LINE} | \
	sed ${IGNORE_COMMENT_LINES} | \
        sed ${DELETE_BRACEPLUS} | \
        sed ${DELETE_PLUSBRACE} | \
        sed ${IGNORE_LINES_WITH_LEADING_SLASH} | \
        sed ${IGNORE_LINES_WITH_LEADING_GREATER_THAN} | \
	sed ${IGNORE_LINES_WITH_LEADING_LEFT_BRACE} | \
        sed ${IGNORE_LINES_WITH_LEADING_RIGHT_BRACE} | \
        sed ${IGNORE_LINES_WITH_RIGHT_BRACKET} | \
        sed ${IGNORE_LINES_THAT_START_WITH_DATA} | \
        sed ${IGNORE_LINES_THAT_START_WITH_TRANSFORMED_DATA} | \
        sed ${IGNORE_LINES_THAT_START_WITH_PARAMETERS} | \
        sed ${IGNORE_LINES_THAT_START_WITH_TRANSFORMED_PARAMETERS} | \
        sed ${IGNORE_LINES_THAT_START_WITH_MODEL} | \
        sed ${IGNORE_LINES_THAT_START_WITH_GENERATED_QUANTITIES} | \
        sed ${IGNORE_LINES_THAT_START_WITH_CODE_COMMENTS} > \
	additions.sh
	echo 'Processing additions to' $TEX_FILE
	exec <additions.sh
	while read -r line; do
		grep  -F -w -n -- "${line}" ${TEX_FILE} > search.txt || true
                UNIQUE_LINES=`wc -l -- search.txt | sed 's/[^[0-9]//g'`
		if  [ $UNIQUE_LINES -eq "1" ]; then
                        LINE_NUM=`sed 's/[^0-9]//g' search.txt`
			sed "${LINE_NUM}s@^${line}@\\\\A{${line}} \\\\FXA \\\\ @" $TEX_FILE > temp.tex
                        mv temp.tex $TEX_FILE
		else
			echo "In ${TEX_FILE}, no unique match for:" "${line}"
		fi
	done
	rm additions.sh
done

MODDED_LINE='\[-.*-\]{+.*+}'
REMOVE_DELETED_PART='s@\[-.*-\]@@g'
DELETE_BRACEPLUS='s@{+@@g'
DELETE_PLUSBRACE='s@+}@@g'

for TEX_FILE in $TEX_FILES; do
        if [ "stan-reference.tex" = "$TEX_FILE" ]; then
                continue
        fi
        git diff --unified=0 --inter-hunk-context=0 --word-diff --color=never $OLD_VERSION $NEW_VERSION $TEX_FILE | \
        grep ${MODDED_LINE} | \
	sed ${REMOVE_DELETED_PART} | \
        sed ${IGNORE_COMMENT_LINES} | \
        sed ${DELETE_BRACEPLUS} | \
        sed ${DELETE_PLUSBRACE} | \
        sed ${IGNORE_LINES_WITH_LEADING_SLASH} | \
        sed ${IGNORE_LINES_WITH_LEADING_GREATER_THAN} | \
        sed ${IGNORE_LINES_WITH_LEADING_LEFT_BRACE} | \
        sed ${IGNORE_LINES_WITH_LEADING_RIGHT_BRACE} | \
        sed ${IGNORE_LINES_WITH_RIGHT_BRACKET} | \
        sed ${IGNORE_LINES_THAT_START_WITH_DATA} | \
        sed ${IGNORE_LINES_THAT_START_WITH_TRANSFORMED_DATA} | \
        sed ${IGNORE_LINES_THAT_START_WITH_PARAMETERS} | \
        sed ${IGNORE_LINES_THAT_START_WITH_TRANSFORMED_PARAMETERS} | \
        sed ${IGNORE_LINES_THAT_START_WITH_MODEL} | \
        sed ${IGNORE_LINES_THAT_START_WITH_GENERATED_QUANTITIES} | \
        sed ${IGNORE_LINES_THAT_START_WITH_CODE_COMMENTS} > \
        changes.sh
        echo 'Processing changes to' $TEX_FILE
        exec <changes.sh
        while read -r line; do
                grep  -F -w -n -- "${line}" ${TEX_FILE} > search.txt || true
                UNIQUE_LINES=`wc -l -- search.txt | sed 's/[^[0-9]//g'`
                if  [ $UNIQUE_LINES -eq "1" ]; then
                        LINE_NUM=`sed 's/[^0-9]//g' search.txt`
                        sed "${LINE_NUM}s@^${line}@\\\\A{${line}} \\\\FXA \\\\ @" $TEX_FILE > temp.tex
                        mv temp.tex $TEX_FILE
                else
                        echo "In ${TEX_FILE}, no unique match for:" "${line}"
                fi
        done
        rm changes.sh
done

rm search.txt
sed 's@^%\\listoffixmes$@\\listoffixmes@' stan-reference.tex > temp.tex
mv temp.tex stan-reference.tex
sed 's@final,author@draft,author@' stan-manuals.sty > temp.sty
sed "s@List of changes@List of changes since ${OLD_VERSION}@" temp.sty > stan-manuals.sty
echo ''
exit 0

