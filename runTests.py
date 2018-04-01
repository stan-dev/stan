#!/usr/bin/python

"""
replacement for runtest target in Makefile
"""

import os
import os.path
import platform
import sys
import subprocess
import time
import imp
mathRunTests = imp.load_source('runTests', 
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "lib", "stan_math", "runTests.py"))

# set up good makefile target name    
def mungeName(name):
    if (name.startswith("src")):
        name = name.replace("src/","",1)
    if (name.endswith(mathRunTests.testsfx)):
        name = name.replace(mathRunTests.testsfx,"")
        if (mathRunTests.isWin()):
            name += mathRunTests.winsfx
            name = name.replace("\\","/")
    return name


def main():
    mathRunTests.mungeName = mungeName
    mathRunTests.main()


if __name__ == "__main__":
    main()
