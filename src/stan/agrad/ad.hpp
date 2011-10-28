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
       * Return a reference to this variable after addition of
       * the specified variable with derivative propagation.
       *
       * @param b Variable summand.
       * @return This plus summand.
       */
      inline 
      fvar<T>& operator+=(const fvar<T>& b) {
	val_ += b.val_;
	prime_ += b.prime_;
	return *this;
      }
      /**
       * Return a reference to this variable after addition of
       * the specified scalar with derivative propagation.
       *
       * @param b Scalar summand.
       * @return This plus summand.
       */
      inline 
      fvar<T>& operator+=(const T& b) {
	val_ += b;
	return *this;
      }

      /**
       * Return a reference to this variable after subtraction
       * of the specified variable with derivative propagation.
       *
       * @param b Variable summand.
       * @return This minus summand.
       */
      inline 
      fvar<T>& operator-=(const fvar<T>& b) {
	val_ -= b.val_;
	prime_ -= b.prime_;
	return *this;
      }
      /**
       * Return a reference to this variable after subtraction of the
       * specified scalar with derivative propagation.
       *
       * @param b Scalar summand.
       * @return This minus summand.
       */
      inline 
      fvar<T>& operator-=(const T& b) {
	val_ -= b;
	return *this;
      }


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


      /**
       * Return a reference to this variable after division by
       * the specified variable with derivative propagation.
       *
       * @param b Variable multiplicand.
       * @return This divided by multiplicand.
       */
      inline 
      fvar<T>& operator/=(const fvar<T>& b) {
	prime_ = (prime_ * b.val_ - val_ * b.prime_) / (b.val_ * b.val_);
	val_ /= b.val_;
	return *this;
      }
      /**
       * Return a reference to this variable after division by
       * the specified scalar with derivative propagation.
       *
       * @param b Scalar multiplicand.
       * @return This divided by multiplicand.
       */
      inline 
      fvar<T>& operator/=(const T& b) {
	val_ /= b;
	prime_ /= b;
	return *this;
      }


    };
      
    /*
     * Equality operator comparing two variable's values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is the same as the
     * second's.
     */
    template <typename T>
    inline 
    bool operator==(const fvar<T>& a, const fvar<T>& b) {
      return a.val() == b.val();
    }
    /**
     * Equality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is the same as the
     * second value.
     */
    template <typename T>
    inline 
    bool operator==(const fvar<T>& a, const double& b) {
      return a.val() == b;
    }
    /**
     * Equality operator comparing a scalar and a variable's value
     * (C++).
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return True if the variable's value is equal to the scalar.
     */
    template <typename T>
    inline 
    bool operator==(const double& a, const fvar<T>& b) {
      return a == b.val();
    }

    /**
     * Inequality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is not the same as the
     * second's.
     */
    template <typename T>
    inline 
    bool operator!=(const fvar<T>& a, const fvar<T>& b) {
      return a.val() != b.val();
    }
    /**
     * Inequality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is not the same as the
     * second value.
     */
    template <typename T>
    inline 
    bool operator!=(const fvar<T>& a, const double& b) {
      return a.val() != b;
    }
    /**
     * Inequality operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable. 
     * @return True if the first value is not the same as the
     * second variable's value.
     */
    template <typename T>
    inline 
    bool operator!=(const double& a, const fvar<T>& b) {
      return a != b.val();
    }

    /**
     * Less than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than second's.
     */
    template <typename T>
    inline 
    bool operator<(const fvar<T>& a, const fvar<T>& b) {
      return a.val() < b.val();
    }
    /**
     * Less than operator comparing variable's value and a double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than second value.
     */
    template <typename T>
    inline 
    bool operator<(const fvar<T>& a, const double& b) {
      return a.val() < b;
    }
    /**
     * Less than operator comparing a double and variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than second variable's value.
     */
    template <typename T>
    inline 
    bool operator<(const double& a, const fvar<T>& b) {
      return a < b.val();
    }

    /**
     * Greater than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than second's.
     */
    template <typename T>
    inline 
    bool operator>(const fvar<T>& a, const fvar<T>& b) {
      return a.val() > b.val();
    }
    /**
     * Greater than operator comparing variable's value and double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than second value.
     */
     template <typename T>
     inline 
     bool operator>(const fvar<T>& a, const double& b) {
       return a.val() > b;
		   }
    /**
     * Greater than operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is greater than second variable's value.
     */
    template <typename T>
    inline 
    bool operator>(const double& a, const fvar<T>& b) {
      return a > b.val();
    }

    /**
     * Less than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than or equal to
     * the second's.
     */
    template <typename T>
    inline 
    bool operator<=(const fvar<T>& a, const fvar<T>& b) {
      return a.val() <= b.val();
    }
    /**
     * Less than or equal operator comparing a variable's value and a
     * scalar (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than or equal to
     * the second value.
     */
    template <typename T>
    inline 
    bool operator<=(const fvar<T>& a, const double& b) {
      return a.val() <= b;
    }
    /**
     * Less than or equal operator comparing a double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than or equal to the second
     * variable's value.
     */
    template <typename T>
    inline 
    bool operator<=(const double& a, const fvar<T>& b) {
      return a <= b.val();
    }

    /**
     * Greater than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than or equal
     * to the second's.
     */
    template <typename T>
    inline 
    bool operator>=(const fvar<T>& a, const fvar<T>& b) {
      return a.val() >= b.val();
    }
    /**
     * Greater than or equal operator comparing variable's value and
     * double (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than or equal
     * to second value.
     */
    template <typename T>
    inline bool operator>=(const fvar<T>& a, const double& b) {
      return a.val() >= b;
    }
    /**
     * Greater than or equal operator comparing double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if the first value is greater than or equal to the
     * second variable's value.
     */
    template <typename T>
    inline bool operator>=(const double& a, const fvar<T>& b) {
      return a >= b.val();
    }

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
    
