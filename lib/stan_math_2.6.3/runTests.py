#!/usr/bin/python

"""
replacement for runtest target in Makefile
arg 1:  test dir or test file
"""

import os
import os.path
import platform
import sys
import subprocess
import time

winsfx = ".exe"
testsfx = "_test.cpp"
debug = False
batchSize = 25

def usage():
    sys.stdout.write('usage: %s <path/test/dir(/files)>\n' % sys.argv[0])
    sys.stdout.write('or\n')
    sys.stdout.write('       %s -j<#cores> <path/test/dir(/files)>\n' % sys.argv[0])
    sys.exit(0)

def stopErr(msg, returncode):
    sys.stderr.write('%s\n' % msg)
    sys.stderr.write('exit now (%s)\n' % time.strftime('%x %X %Z'))
    sys.exit(returncode)

def isWin():
    if (platform.system().lower().startswith("windows")
        or os.name.lower().startswith("windows")):
        return True
    return False

# set up good makefile target name    
def mungeName(name):
    if (name.endswith(testsfx)):
        name = name.replace(testsfx,"_test")
        if (isWin()):
            name += winsfx
            name = name.replace("\\","/")
    return name

def doCommand(command):
    print("------------------------------------------------------------")
    print("%s" % command)
    p1 = subprocess.Popen(command,shell=True)
    p1.wait()
    if (not(p1.returncode == None) and not(p1.returncode == 0)):
        stopErr('%s failed' % command, p1.returncode)

def generateTests(j):
    if (j == None):
        command = 'make generate-tests -s'
    else:
        command = 'make -j%d generate-tests -s' % j
    doCommand(command)

def makeTest(name, j):
    target = mungeName(name)
    if (j == None):
        command = 'make %s' % target
    else:
        command = 'make -j%d %s' % (j,target)
    doCommand(command)
    
def makeTests(dirname, filenames, j):
    targets = list()
    for name in filenames:
        if (not name.endswith(testsfx)):
            continue
        target = "/".join([dirname,name])
        target = mungeName(target)
        targets.append(target)
    if (len(targets) > 0):
        if (debug):
            print('# targets: %d' % len(targets))
        startIdx = 0
        endIdx = batchSize
        while (startIdx < len(targets)):
            if (j == None):
                command = 'make %s' % ' '.join(targets[startIdx:endIdx])
            else:
                command = 'make -j%d %s' % (j,' '.join(targets[startIdx:endIdx]))
            if (debug):
                print('start %d, end %d' % (startIdx,endIdx))
                print(command)
            doCommand(command)
            startIdx = endIdx
            endIdx = startIdx + batchSize
            if (endIdx > len(targets)):
                endIdx = len(targets)
         

def runTest(name):
    executable = mungeName(name).replace("/",os.sep)
    xml = mungeName(name).replace(winsfx, "")
    command = '%s --gtest_output="xml:%s.xml"' % (executable, xml)
    doCommand(command)

def main():
    if (len(sys.argv) < 2):
        usage()

    argsIdx = 1
    j = None
    if (sys.argv[1].startswith("-j")):
        argsIdx = 2
        if (len(sys.argv) < 3):
            usage()
        else:
            j = sys.argv[1].replace("-j","")
            try:
                jprime = int(j)
                if (jprime < 1 or jprime > 16):
                    stopErr("bad value for -j flag",-1)                    
                j = jprime
            except ValueError:
                stopErr("bad value for -j flag",-1)
            
    # pass 0: generate all auto-generated tests
    generateTests(j)

    # pass 1:  call make to compile test targets
    for i in range(argsIdx,len(sys.argv)):
        testname = sys.argv[i]
        if (not(os.path.exists(testname))):
            stopErr('%s: no such file or directory' % testname,-1)
        if (not(os.path.isdir(testname))):
            if (not(testname.endswith(testsfx))):
                stopErr('%s: not a testfile' % testname,-1)
            if (debug):
                print("make single test: %s" % testname)
            makeTest(testname,j)
        else:
            for root, dirs, files in os.walk(testname):
                if (debug):
                    print("make root: %s" % root)
                makeTests(root,files,j)

    # pass 2:  run test targets
    for i in range(argsIdx,len(sys.argv)):
        testname = sys.argv[i]
        if (not(os.path.isdir(testname))):
            if (debug):
                print("run single test: %s" % testname)
            runTest(testname)
        else:
            for root, dirs, files in os.walk(testname):
                for name in files:
                    if (name.endswith(testsfx)):
                        if (debug):
                            print("run dir,test: %s,%s" % (root,name))
                        runTest(os.sep.join([root,name]))

    
if __name__ == "__main__":
    main()
