Combining Inclusion Model & Explicit Instantiation
==================================================

This example follows the description in:

  Vandevoorde, David and Nicolai M. Josuttis.  2003. 
  C++ Templates: the Complete Guide.  Addison-Wesley.
  p. 67.


LIBRARY ORGANIZATION
----------------------------------------------------------------------

To separate the declaration and definition, organize
function (or class) definitions into two files:

  foo.hpp
  * declare template function foo()
 
  foo_def.hpp
  * include foo.hpp
  * define template function foo()


INCLUSION MODEL
----------------------------------------------------------------------

For header-only use in the inclusion model, include
the definition file.

  incl_demo.cpp
  * include foo_def.hpp
  * call foo<double> in main()

To compile and run:

  > clang++ incl_demo.cpp -o incl_demo
  > ./incl_demo
  Inclusion model. foo(3.0)=6
  >


EXPLICIT INSTANTIATIATION MODEL
----------------------------------------------------------------------

The goal is to break compilation into separate translation
units.

The first translation unit contains the explicit
instantiations of all template instances that will be
needed to link the second translation unit.

  foo_inst.cpp
  * include foo_def.hpp
  * explicitly instantiate foo<double>

The second translation unit is the command.  It only includes
the header declaration file.  

  instantiation_model.cpp
  * includes foo.hpp
  * defines main() function
  * calls foo<double>

Compile the two translation units, link, and run:

  > clang++ -c inst_demo.cpp
  > clang++ -c foo_inst.cpp
  > clang++ inst_demo.o foo_inst.o -o inst_demo
  > ./run_foo
