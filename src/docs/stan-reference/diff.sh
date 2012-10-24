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

TEX_FILES=`ls *.tex`
#TEX_FILES=functions.tex
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
        sed ${IGNORE_LINES_WITH_LEADING_SLASH} > \
	additions.sh
	echo 'Processing additions to' $TEX_FILE
	exec <additions.sh
	while read -r line; do
		UNIQUE_LINES=`grep  -F -w "${line}" ${TEX_FILE} | wc -l`
		if  [ $UNIQUE_LINES -eq "1" ]; then
#			replace -s "${line}" "\A{$line} \FXA " -- $TEX_FILE
			sed -i "s@^${line}@ \\\\A{${line}} \\\\FXA @" $TEX_FILE
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
        sed ${IGNORE_LINES_WITH_LEADING_SLASH} > \
        changes.sh
        echo 'Processing changes to' $TEX_FILE
        exec <changes.sh
        while read -r line; do
                UNIQUE_LINES=`grep -F -w "${line}" ${TEX_FILE} | wc -l`
                if  [ $UNIQUE_LINES -eq "1" ]; then
#                        replace -s "${line}" "\A{$line} \FXC " -- $TEX_FILE
			sed -i "s@^${line}@ \\\\A{${line}} \\\\FXC@ " $TEX_FILE
                else
                        echo "In ${TEX_FILE}, no unique match for:" "${line}"
                fi
        done
        rm changes.sh
done

sed -i 's@^%\\listoffixmes$@\\listoffixmes@' stan-reference.tex
sed -i 's@final,author@draft,author@' stan-manuals.sty
sed -i "s@List of changes@List of changes since ${OLD_VERSION}@" stan-manuals.sty
echo ''
echo 'Steps from here:'
echo '1) Apply --- but do not commit --- patches (if any)'
echo '2) make manual'
echo '3) git reset --hard HEAD'
