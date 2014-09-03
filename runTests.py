#!/usr/bin/python

import os
import os.path
import re
import sys
import subprocess
import time

"""
replacement for runtest target in Makefile
arg 1:  test dir or test file
"""

def stop_err( msg ):
    sys.stderr.write( '%s\n' % msg )
    sys.stderr.write( 'exit now ( %s)\n' % time.strftime('%x %X %Z'))
    sys.exit()


#       p1 = subprocess.Popen(command,shell=True)
#,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#    (sout,serr) = p1.communicate()
#    if (sout != None):
#        sys.stdout.write(sout)
#    if (serr != None):
#        sys.stderr.write(serr)
#                
#    if(not(p1.returncode == None) and not(p1.returncode == 0)):
#        stop_err("annotation GUI error")
def domake( target ):
    print("make",target)
    
def dotest( target ):
    print("do",target)
    

    

#test/unit-distribution/generate_tests$(EXE) : src/test/unit-distribution/generate_tests.cpp
#	@mkdir -p $(dir $@)
#	$(LINK.c) -O$(O_STANC) $(CFLAGS) $< $(OUTPUT_OPTION)
#
#src/test/unit-distribution/%_00000_generated_test.cpp : src/test/unit-distribution/%_test.hpp | test/unit-distribution/generate_tests$(EXE)
#	@echo "--- Generating tests for $(notdir $<) ---"
#	$(WINE) test/unit-distribution/generate_tests$(EXE) $<
def gentests( filename ):    
    print("gentests:",filename)
    # a. make test generator 
    # b. generate tests
    # c. call run1test on each generated test
    
def run1test( filename ):
    target = re.sub("_test.cpp","",filename)
    print("run1test",target)
    domake(target)
    dotest(target)
    
def main():
    testsfx = "_test.cpp"
    pathsep = '/'
    print("cwd: ",os.getcwd())

    name = sys.argv[1]
    print("testing",name)
    testpath = pathsep.join(["src","test",name])
    print(testpath)
    if (not(os.path.exists(testpath))):
        stop_err( '%s: no such file or directory' % testpath)
    if (not(os.path.isdir(testpath))):
        if (not(testpath.endswith(testsfx))):
            stop_err( '%s: not a testfile' % testpath)
        run1test(testpath)
    else:
        for root, dirs, files in os.walk(testpath):
            if ("unit-distribution" in root):
                gentests(root)
            else:
                print(root)
                for name in files:
                    if (name.endswith(testsfx)):
                        run1test(pathsep.join([root,name]))


    
if __name__ == "__main__":
    main()

