#!/bin/bash

# from http://stackoverflow.com/questions/1404796/how-to-get-the-latest-tag-name-in-current-branch-in-git
PreviousAndCurrentGitTag=`git describe --tags \`git rev-list --tags --abbrev=0 --max-count=2\` --abbrev=0`
PreviousGitTag=`echo $PreviousAndCurrentGitTag | cut -f 2 -d ' '`
CurrentGitTag=`echo $PreviousAndCurrentGitTag | cut -f 1 -d ' '`

OLDVERSION=$1
if [ -z "$OLDVERSION" ]; then
        OLDVERSION=$PreviousGitTag
fi
NEWVERSION=$2
if [ -z "$NEWVERSION" ]; then
        NEWVERSION=$CurrentGitTag
fi

BRANCH=`git branch | grep ^*`
if [ "* master" != "$BRANCH" ]; then
        echo 'You must be on the master branch to run this script'
        exit 3
fi

if ! git diff-index --quiet HEAD --; then
        echo "You have uncommited changes that need to be committed or stashed before running this script; aborting"
        exit 4
fi

STAN_HOME=`pwd`
cd $STAN_HOME/src/docs/stan-reference
bash diff.sh $OLDVERSION $NEWVERSION
if [ $? != 0 ]; then
        echo "Diffing failed, aborting"
        git reset --hard HEAD
        git checkout master
        exit 5
fi
cd $STAN_HOME
if [ -e ${STAN_HOME}/src/docs/stan-reference/diff_patches/${OLDVERSION}_${NEWVERSION}.patch ]; then
        git apply ${STAN_HOME}/src/docs/stan-reference/diff_patches/${OLDVERSION}_${NEWVERSION}.patch
        if [ $? != 0 ]; then
                echo "Applying patch failed, aborting"
                git reset --hard HEAD
                git checkout master
                exit 6
        fi
fi
make clean-all && make manual
if [ $? != 0 ]; then
        echo "Creating diffed manual failed, aborting"
        git reset --hard HEAD
        git checkout master
        exit 7
fi

git reset --hard HEAD
git checkout master
echo "Manual diffed successfully (in theory); you can upload stan-reference-*.pdf"
exit 0
