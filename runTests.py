#!/usr/bin/python

import argparse
import os
import os.path
import platform
import re
import sys
import subprocess
import time

"""
replacement for runtest target in Makefile
arg 1:  test dir or test file
"""

winsfx = ".exe"
testsfx = "_test.cpp"

def usage():
    sys.stdout.write('usage: %s <path/test/dir(/files)>\n' % sys.argv[0])
    sys.stdout.write('or\n')
    sys.stdout.write('       %s -j<#cores> <path/test/dir(/files)>\n' % sys.argv[0])
    sys.exit(0)

def stop_err( msg, returncode ):
    sys.stderr.write( '%s\n' % msg )
    sys.stderr.write( 'exit now ( %s)\n' % time.strftime('%x %X %Z'))
    sys.exit(returncode)

def doCommand(command):
    p1 = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (sout,serr) = p1.communicate()
    if (sout != None):
        sys.stdout.write(sout.decode())
    if (serr != None):
        sys.stderr.write(serr.decode())
    if (not(p1.returncode == None) and not(p1.returncode == 0)):
        stop_err('%s failed' % command, p1.returncode)

def generateTests(j):
    command = 'make -j%d generate-tests -s' % j
    print(command)
    doCommand(command)

def makeTest( name, j ):
    name = name.replace("src/","",1)
    name = name.replace(testsfx,"");
    command = 'make -j%d %s' % (j,name)
    doCommand(command)
    
def makeTests( dirname, filenames, j ):
    dirname = dirname.replace("src/","",1)
    targets = list()
    for name in filenames:
        if (name.endswith(testsfx)):
            target = name.replace(testsfx,"");
            if (isWin()):
                target += winsfx
                sys.out.println('windows target name: %s' % os.sep.join([dirname,target]))
            targets.append(os.sep.join([dirname,target]))
    if (len(targets) > 0):
        command = 'make -j%d %s' % (j,' '.join(targets))
        print(command)
        doCommand(command)
    
def runTest( name ):
    name = name.replace("src/","",1)
    name = name.replace(testsfx,"");
    print(name)
    doCommand(name)

def isWin():
    if (platform.system().lower().startswith("windows")
        or os.name.lower().startswith("windows")):
        return True
    return False


def main():
    if (len(sys.argv) < 2):
        usage()

    j = 1
    start = 1
    if (sys.argv[1].startswith("-j")):
        start = 2
        if (len(sys.argv) < 3):
            usage()
        else:
            j = sys.argv[1].replace("-j","")
            try:
                jprime = int(j)
                if (jprime < 1 or jprime > 16):
                    stop_err("bad value for -j flag",-1)                    
                j = jprime
            except ValueError:
                stop_err("bad value for -j flag"-1)
            
    print("j:",j,"start",start)

    # pass 0: generate all auto-generated tests
    generateTests(j)

    # pass 1:  call make to compile test targets
    for i in range(start,len(sys.argv)):
        testname = sys.argv[i]
        if (not(os.path.exists(testname))):
            stop_err( '%s: no such file or directory' % testname,-1)
        if (not(os.path.isdir(testname))):
            if (not(testname.endswith(testsfx))):
                stop_err( '%s: not a testfile' % testname,-1)
            makeTest(testname,j)
        else:
            for root, dirs, files in os.walk(testname):
                makeTests(root,files,j)

    # pass 2:  run test targets
    for i in range(start,len(sys.argv)):
        testname = sys.argv[i]
        if (not(os.path.isdir(testname))):
            testexe = testname.replace(testsfx,"");
            runTest(testexe)
        else:
            for root, dirs, files in os.walk(testname):
                for name in files:
                        if (name.endswith(testsfx)):
                            testexe = name.replace(testsfx,"");
                            runTest(os.sep.join([root,testexe]))


    
if __name__ == "__main__":
    main()

