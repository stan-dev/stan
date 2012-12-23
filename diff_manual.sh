#!/bin/bash

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
./diff.sh $1 $2
if [ $? != 0 ]; then
        echo "Diffing failed, aborting"
        git reset --hard HEAD
        git checkout master
        exit 5
fi
cd $STAN_HOME
git apply $STAN_HOME/src/docs/stan-reference/diff_patches/$1_$2.patch
if [ $? != 0 ]; then
        echo "Applying patch failed, aborting"
        git reset --hard HEAD
        git checkout master
        exit 6
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
