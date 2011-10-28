#ifndef __STAN__AGRAD__AD_HPP__
#define __STAN__AGRAD__AD_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

namespace stan {

  namespace agrad {

    /**
     * The variable class used for forward-mode algorithmic
     * differentiation.
     * 
     * @tparam T Type of scalar values.
     */
    template <typename T>
    class fvar {
      T val_;
      T prime_;

    public:

      /**
       * Construct a forward-mode algorithmic differentiation
       * variable with the specified value.  The variable's
       * derivative will be set to 0, making <code>fvar(x)</code>
       * shorthand for <code>fvar(x,0.0)</code>.
       *
       * <p>In order for this to work, the class scalar type
       * <code>T</code> must be constructible with an instance of 
       * the constructor scalar type <code>S</code>.
       *
       * @tparam S Type of scalar value.
       * @param val Value of constructed variable.
       */
      template <typename S>
      fvar(S val) : val_(val), prime_(0.0) { }

      /**
       * Construct a forward-mode algorithmic differentiation variable
       * with the specified value and derivative with respect to the
       * distinguished independent variable.
       *
       * <p>An instance of the class scalar type <code>T</code> must
       * be constructible with an instance of the constructor scalar
       * type <code>S</code>.
       *
       * @tparam S Type of scalar value.
       * @param val Value of constructed variable.       * @param prime Value of the derivative.
       */
      template <typename S>
      fvar(S val, S prime) : val_(val), prime_(prime) { }

      /**
       * Return the value of this variable.
       *
       * @return Value of this variable.
       */
      const T val() const { return val_; }

      /**
       * Return the derivative of this variable with respect to the
       * independent variable.
       *
       * @return Derivative of this variable.
       */
      const T prime() const { return prime_; }
      
      /**
       * Return a reference to this variable after multiplication by
       * the specified variable with derivative propagation.
       *
       * @param b Variable multiplicand.
       * @return This times multiplicand.
       */
      inline 
      fvar<T>& operator*=(const fvar<T>& b) {
	prime_ = prime_ * b.val_ + val_ * b.prime_;
	val_ *= b.val_;
	return *this;
      }

      /**
       * Return a reference to this variable after multiplication by
       * the specified scalar with derivative propagation.
       *
       * @param b Scalar multiplicand.
       * @return This times multiplicand.
       */
      inline 
      fvar<T>& operator*=(const T& b) {
	val_ *= b;
	prime_ *= b;
	return *this;
      }


    };

    /**
     * The class for representing a distinguished independent variable
     * with respect to which forward-mode algorithmic differentiation
     * is carried out.
     *
     * <p>This class is really just a convenience method for the
     * superclass, and instances may thus be sliced without fear of
     * information loss.
     *
     * @tparam T Type of scalar.
     */
    template <typename T>
    class indep_fvar : public fvar<T> {
    public:
      /**
       * Construct the independent forward-mode algorithmic
       * differentiation variable with respect to which algorithmic
       * differentiation is carried out.  
       *
       * <p>An instance of class scalar type <code>T</code> must be
       * constructible from an instance of the constructor scalar
       * type <code>S</code>.
       *
       * @tparam S Type of argument scalar; <code>T</code> is the
       * type of the constructed scalar class.
       * @param val Value of the variable.
       */
      template <typename S>
      indep_fvar(S val) : fvar<T>(val,S(1.0)) { }
    };

    /**
     * Return the product (and derivative) of the two variables.
     *
     * @param a First variable.
     * @param b Second variable.
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator*(const fvar<T>& a, const fvar<T>& b) {
      return fvar<T>(a.val() * b.val(), a.prime() * b.val() + a.val() * b.prime());
    }

    /**
     * Return the product (and derivative) of the variable and
     * scalar.
     *
     * @param a First variable.
     * @param b Second scalar.
     * @return Product of the variable and scalar.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>  
    inline 
    fvar<T> operator*(const fvar<T>& a, const T& bf) {
      return fvar<T>(a.val() * bf, a.prime() * bf);
    }

    /**
     * Return the product (and derivative) of the scalar and variable.
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return Product of the scalar and variable.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator*(const T& af, const fvar<T>& b) {
      return fvar<T>(af * b.val(), af * b.prime());
    }
  }

}

#endif
    
