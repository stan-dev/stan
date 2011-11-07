#ifndef __STAN__AGRAD__AD_HPP__
#define __STAN__AGRAD__AD_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

namespace stan {

  namespace agrad {

    using std::exp;
    using std::log;
    using std::log10;

    /**
     * The variable class used for forward-mode algorithmic
     * differentiation.
     * 
     * @tparam T Type of scalar values.
     */
    template <typename T>
    class fvar {
    public:

      /**
       * The value of this variable.
       */
      T val_;

      /**
       * The derivative of this variable with respect to the
       * distinguished independent variable.
       */
      T prime_;


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
       * @param val Value of constructed variable.      
       * @param prime Value of the derivative.
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


      // COMPOUND ASSIGNMENT OPERATORS

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
      fvar<T>& operator+=(const double& b) {
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
      fvar<T>& operator-=(const double& b) {
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
      fvar<T>& operator*=(const double& b) {
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
      fvar<T>& operator/=(const double& b) {
	val_ /= b;
	prime_ /= b;
	return *this;
      }


    };

    // COMPARISON OPERATORS
      
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
     * Less than operator comparing variable's value and a scalar
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
     * Less than operator comparing a scalar and variable value
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
     * Greater than operator comparing variable's value and double.
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


    // LOGICAL OPERATORS

    /**
     * Prefix logical negation for the value of variables (C++).  The
     * expression (!a) is equivalent to negating the scalar value of
     * the variable a.
     *
     * Note that this is the only logical operator defined for
     * variables.  Overridden logical conjunction (&&) and disjunction
     * (||) operators do not apply the same "short circuit" rules
     * as the built-in logical operators.  
     *
     * @param a Variable to negate.
     * @return True if variable is non-zero.
     */
     template <typename T>
     inline bool operator!(const fvar<T>& a) {
       return !a.val();
     }


    // ARITHMETIC OPERATORS

    /**
     * Unary plus operator for variables (C++).  
     *
     * <p>The function simply returns its input.  The effect of unary
     * plus on a built-in C++ scalar type is integer promotion.
     * Because variables are all double-precision floating point
     * already, promotion is not necessary.
     *
     * \f$(+f)' = +f' = f'\f$.
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    template <typename T>
    inline fvar<T> operator+(const fvar<T>& a) {
      return a;
    }

    /**
     * Unary minus operator for variables (C++).  
     *
     * \f$(-f)' = -(f')\f$.
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    template <typename T>
    inline fvar<T> operator-(const fvar<T>& a) {
      return fvar<T>(-a.val(),-a.prime());
    }

    /**
     * Return the sum (and derivative) of the two variables.
     *
     * \f$(f + g)' = f' + g'\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator+(const fvar<T>& a, const fvar<T>& b) {
      return fvar<T>(a.val() + b.val(), a.prime() + b.prime());
    }

    /**
     * Return the sum (and derivative) of the variable and
     * scalar.
     *
     * \f$(f + c)' = f'\f$.
     *
     * @param a First variable.
     * @param bf Second scalar.
     * @return Product of the variable and scalar.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>  
    inline 
    fvar<T> operator+(const fvar<T>& a, const double& bf) {
      return fvar<T>(a.val() + bf,  a.prime());
    }

    /**
     * Return the product (and derivative) of the scalar and variable.
     *
     * \f$(c + g)' = g'\f$.
     * 
     * @param af First scalar.
     * @param b Second variable.
     * @return Product of the scalar and variable.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator+(const double& af, const fvar<T>& b) {
      return fvar<T>(af + b.val(), b.prime());
    }



    /**
     * Return the difference (and derivative) of the two variables.
     *
     * \f$(f - g)' = f' - g'\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return Difference of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator-(const fvar<T>& a, const fvar<T>& b) {
      return fvar<T>(a.val() - b.val(), a.prime() - b.prime());
    }

    /**
     * Return the difference (and derivative) between the variable and
     * scalar.
     *
     * \f$(f - c)' = f'\f$.
     *
     * @param a First variable.
     * @param bf Second scalar.
     * @return Difference of the variable and scalar.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>  
    inline 
    fvar<T> operator-(const fvar<T>& a, const double& bf) {
      return fvar<T>(a.val() - bf,  a.prime());
    }

    /**
     * Return the difference (and derivative) between the scalar and
     * variable.
     *
     * \f$(c - g)' = -g'\f$.
     * 
     * @param af First scalar.
     * @param b Second variable.
     * @return Difference of the scalar and variable.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator-(const double& af, const fvar<T>& b) {
      return fvar<T>(af - b.val(), -b.prime());
    }

    /**
     * Return the product (and derivative) of the two variables.
     *
     * \f$(fg)' = f'g + fg'\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator*(const fvar<T>& a, const fvar<T>& b) {
      return fvar<T>(a.val() * b.val(),
		     a.prime() * b.val() + a.val() * b.prime());
    }

    /**
     * Return the product (and derivative) of the variable and
     * scalar.
     *
     * \f$(fc)' = f'c\f$.
     *
     * @param a First variable.
     * @param bf Second scalar.
     * @return Product of the variable and scalar.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>  
    inline 
    fvar<T> operator*(const fvar<T>& a, const double& bf) {
      return fvar<T>(a.val() * bf, a.prime() * bf);
    }

    /**
     * Return the product (and derivative) of the scalar and variable.
     *
     * \f$(cg)' = cg'\f$.
     * 
     * @param af First scalar.
     * @param b Second variable.
     * @return Product of the scalar and variable.
     * @tparam T Type of scalar in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator*(const double& af, const fvar<T>& b) {
      return fvar<T>(af * b.val(), af * b.prime());
    }


    /**
     * Return the division of the first variable by the second.
     *
     * \f$(f/g)' = (f'g - fg')/g^2\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator/(const fvar<T>& a, const fvar<T>& b) {
      return fvar<T>(a.val() / b.val(),
		     (a.prime() * b.val() - a.val() * b.prime())
		     / (b.val() * b.val()));
    }
    
    /**
     * Return the division of the first variable by the second scalar.
     *
     * \f$(f/c)' = f'/c\f$.
     * 
     * @param a First variable.
     * @param b Second scalar
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator/(const fvar<T>& a, const double& b) {
      return fvar<T>(a.val() / b, a.prime() / b);
    }
    
    /**
     * Return the division of the first scalar by the second variable.
     *
     * \f$(c/g)' = -cg' / g^2\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return Product of the variables.
     * @tparam T Type of scalars in variables.
     */
    template <typename T>
    inline 
    fvar<T> operator/(const double& a, const fvar<T>& b) {
      return fvar<T>(a / b.val(),
		     - (a * b.prime())
		     / (b.val() * b.val()));
    }

    /**
     * Prefix increment operator for variables (C++).
     * 
     * Following C++, the expression <code>(a++)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>a = a + 1.0;  return a;</code>
     *
     * @param a Variable to increment.
     * @return Input variable incremented. 
     */
    template <typename T>
    inline
    fvar<T> operator++(fvar<T>& a) {
      ++a.val_;
      return a;
    }

    /**
     * Postfix increment operator for variables (C++).
     * 
     * Following C++, the expression <code>(a++)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>fvar<T> temp = a;  a = a + 1.0;  return temp;</code>
     *
     * @param a Variable to increment.
     * @param dummy Unused dummy variable used to distinguish postfix operator
     * from prefix operator.
     * @return Input variable. 
     */
    template <typename T>
    inline
    fvar<T> operator++(fvar<T>& a, int dummy) {
      fvar<T> temp(a.val_,a.prime_);
      ++a.val_;
      return temp;
    }

    /**
     * Prefix decrement operator for variables (C++).
     * 
     * Following C++, the expression <code>(a--)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>a = a - 1.0;  return a;</code>
     *
     * @param a Variable to decrement.
     * @return Decremented input.
     */
    template <typename T>
    inline
    fvar<T> operator--(fvar<T>& a) {
      --a.val_;
      return a;
    }

    /**
     * Postfix decrement operator for variables (C++).
     * 
     * Following C++, the expression <code>(a--)</code> is defined to
     * behave like the sequence of operations
     *
     * <code>fvar<T> temp = a;  a = a - 1.0;  return temp;</code>
     *
     * @param a Variable to decrement.
     * @param dummy Unused dummy variable used to distinguish postfix operator
     * from prefix operator.
     * @return Input variable. 
     */
    template <typename T>
    inline
    fvar<T> operator--(fvar<T>& a, int dummy) {
      fvar<T> temp(a.val_,a.prime_);
      --a.val_;
      return temp;
    }

    // CMATH EXP AND LOG

    /**
     * Return the exponentiation of the specified variable (cmath).
     *
     * \f$\exp'(x) = \exp(x)\f$.
     * 
     * @param a Variable to exponentiate.
     * @return Exponentiated value.
     * @tparam Scalar type.
     */
    template <typename T>
    inline
    fvar<T> exp(const fvar<T>& a) {
      T exp_a = exp(a.val());
      return fvar<T>(exp_a, a.prime() * exp_a);
    }

    /**
     * Return the natural logarithm of the specified variable (cmath).
     *
     * \f$\log'(x) = 1/x\f$.
     * 
     * @param a Variable argument.
     * @return Logarithm of the argument.
     * @tparam Scalar type.
     */
    template <typename T>
    inline
    fvar<T> log(const fvar<T>& a) {
      return fvar<T>(log(a.val_), 1.0/a.val_ * a.prime_);
    }

    namespace {
      const double ONE_OVER_LOG_10 = 1.0 / log(10.0);
    }

    /**
     * Return the base 10 logarithm of the specified variable (cmath).
     *
     * \f$\log_{10}'(x) = 1/(x \log 10)\f$.
     * 
     * @param a Variable argument.
     * @return Base 10 logarithm of the argument.
     * @tparam Scalar type.
     */
    template <typename T>
    inline
    fvar<T> log10(const fvar<T>& a) {
      return fvar<T>(log10(a.val_), a.prime_ * ONE_OVER_LOG_10 / a.val_);
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

  }
}
#endif
    
