#ifndef STAN__AGRAD__REV__VAR_HPP
#define STAN__AGRAD__REV__VAR_HPP

#include <ostream>
#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {

    // forward declare
    static void grad(chainable* vi);
    
    /**
     * Independent (input) and dependent (output) variables for gradients.
     *
     * This class acts as a smart pointer, with resources managed by
     * an agenda-based memory manager scoped to a single gradient
     * calculation.
     *
     * An agrad::var is constructed with a double and used like any
     * other scalar.  Arithmetical functions like negation, addition,
     * and subtraction, as well as a range of mathematical functions
     * like exponentiation and powers are overridden to operate on
     * agrad::var values objects.
     */
    class var {
    public:

      // FIXME: doc what this is for
      typedef double Scalar;

      /**
       * Pointer to the implementation of this variable.  
       *
       * This value should not be modified, but may be accessed in
       * <code>var</code> operators to construct <code>vari</code>
       * instances.
       */
      vari * vi_;

      /**
       * Return <code>true</code> if this variable has been
       * declared, but not been defined.  Any attempt to use an
       * undefined variable's value or adjoint will result in a
       * segmentation fault.
       *
       * @return <code>true</code> if this variable does not yet have
       * a defined variable.
       */ 
      bool is_uninitialized() {
        return (vi_ == static_cast<vari*>(0U));
      }

      /**
       * Construct a variable from a pointer to a variable implementation.
       *
       * @param vi Variable implementation. 
       */
      explicit var(vari* vi)
        : vi_(vi) 
      {  }

      /**
       * Construct a variable for later assignment.   
       *
       * This is implemented as a no-op, leaving the underlying implementation
       * dangling.  Before an assignment, the behavior is thus undefined just
       * as for a basic double.
       */
      var() 
        : vi_(static_cast<vari*>(0U))
      { }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param b Value.
       */
      var(bool b) :
        vi_(new vari(static_cast<double>(b))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param c Value.
       */
      var(char c) :
        vi_(new vari(static_cast<double>(c))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(short n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned short n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(long int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned long int n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(unsigned long long n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param n Value.
       */
      var(long long n) :
        vi_(new vari(static_cast<double>(n))) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param x Value.
       */
      var(float x) :
        vi_(new vari(static_cast<double>(x))) {
      }

      /**      
       * Construct a variable with the specified value.
       *
       * @param x Value of the variable.
       */
      var(double x) :
        vi_(new vari(x)) {
      }

      /**      
       * Construct a variable by static casting the specified
       * value to <code>double</code>.
       *
       * @param x Value.
       */
      var(long double x) :
        vi_(new vari(static_cast<double>(x))) {
      }

      /**
       * Return the value of this variable.
       *
       * @return The value of this variable.
       */
      inline double val() const {
        return vi_->val_;
      }

      /**
       * Return the derivative of the root expression with
       * respect to this expression.  This method only works
       * after one of the <code>grad()</code> methods has been
       * called.  
       *
       * @return Adjoint value for this variable.
       */
      inline double adj() const {
        return vi_->adj_;
      }

      /**
       * Compute the gradient of this (dependent) variable with respect to
       * the specified vector of (independent) variables, assigning the
       * specified vector to the gradient.
       *
       * The grad() function does <i>not</i> recover memory.  In Stan
       * 2.4 and earlier, this function did recover memory.
       *
       * @param x Vector of independent variables.
       * @param g Gradient vector of partial derivatives of this
       * variable with respect to x.
       */
      void grad(std::vector<var>& x,
                std::vector<double>& g) {
        stan::agrad::grad(vi_); // defined in chainable.hpp
        g.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) 
          g[i] = x[i].vi_->adj_;
      }
      
      // POINTER OVERRIDES
      
      /**
       * Return a reference to underlying implementation of this variable.
       *
       * If <code>x</code> is of type <code>var</code>, then applying
       * this operator, <code>*x</code>, has the same behavior as
       * <code>*(x.vi_)</code>.
       *
       * <i>Warning</i>:  The returned reference does not track changes to
       * this variable.
       *
       * @return variable
       */
      inline vari& operator*() {
        return *vi_;
      }

      /**
       * Return a pointer to the underlying implementation of this variable.
       *
       * If <code>x</code> is of type <code>var</code>, then applying
       * this operator, <code>x-&gt;</code>, behaves the same way as
       * <code>x.vi_-&gt;</code>.
       *
       * <i>Warning</i>: The returned result does not track changes to
       * this variable.
       */
      inline vari* operator->() {
        return vi_;
      }

      // COMPOUND ASSIGNMENT OPERATORS
    
      /**
       * The compound add/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a += b) behaves exactly the same way as (a = a + b),
       * creating an intermediate variable representing (a + b).
       * 
       * @param b The variable to add to this variable.
       * @return The result of adding the specified variable to this variable.
       */
      inline var& operator+=(const var& b); 

      /**
       * The compound add/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a += b) behaves exactly the same way as (a = a + b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to add to this variable.
       * @return The result of adding the specified variable to this variable.
       */
      inline var& operator+=(const double b);

      /**
       * The compound subtract/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a -= b) behaves exactly the same way as (a = a - b).
       * Note that the result is an assignable lvalue.
       *
       * @param b The variable to subtract from this variable.
       * @return The result of subtracting the specified variable from
       * this variable.
       */
      inline var& operator-=(const var& b);

      /**
       * The compound subtract/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a -= b) behaves exactly the same way as (a = a - b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to subtract from this variable.
       * @return The result of subtracting the specified variable from this
       * variable.
       */
      inline var& operator-=(const double b);

      /**
       * The compound multiply/assignment operator for variables (C++).  
       *
       * If this variable is a and the argument is the variable b,
       * then (a *= b) behaves exactly the same way as (a = a * b).
       * Note that the result is an assignable lvalue.
       *
       * @param b The variable to multiply this variable by.
       * @return The result of multiplying this variable by the
       * specified variable.
       */
      inline var& operator*=(const var& b);

      /**
       * The compound multiply/assignment operator for scalars (C++).  
       *
       * If this variable is a and the argument is the scalar b, then
       * (a *= b) behaves exactly the same way as (a = a * b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to multiply this variable by.
       * @return The result of multplying this variable by the specified
       * variable.
       */
      inline var& operator*=(const double b);

      /**
       * The compound divide/assignment operator for variables (C++).  If this
       * variable is a and the argument is the variable b, then (a /= b)
       * behaves exactly the same way as (a = a / b).  Note that the
       * result is an assignable lvalue.
       *
       * @param b The variable to divide this variable by.
       * @return The result of dividing this variable by the
       * specified variable.
       */
      inline var& operator/=(const var& b);

      /**
       * The compound divide/assignment operator for scalars (C++). 
       *
       * If this variable is a and the argument is the scalar b, then
       * (a /= b) behaves exactly the same way as (a = a / b).  Note
       * that the result is an assignable lvalue.
       *
       * @param b The scalar to divide this variable by.
       * @return The result of dividing this variable by the specified
       * variable.
       */
      inline var& operator/=(const double b);

      /**
       * Write the value of this auto-dif variable and its adjoint to 
       * the specified output stream.
       *
       * @param os Output stream to which to write.
       * @param v Variable to write.
       * @return Reference to the specified output stream.
       */
      friend std::ostream& operator<<(std::ostream& os, const var& v) {
        if (v.vi_ == 0)
          return os << "uninitialized";
        return os << v.val() << ':' << v.adj();
      }
    };

  }
}
#endif
