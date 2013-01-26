#ifndef __STAN__AGRAD__FVAR_HPP__
#define __STAN__AGRAD__FVAR_HPP__

#include <boost/math/special_functions/cbrt.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    struct fvar {

      T val_;  // value
      T d_;    // tangent (aka derivative)

      // TV and TD must be assignable to T
      template <typename TV, typename TD>
      fvar(const TV& val, const TD& deriv) : val_(val), d_(deriv) {  }

      // TV must be assignable to T
      template <typename TV>
      fvar(const TV& val) : val_(val), d_(0.0) {  }
      
      fvar() : val_(0.0), d_(0.0) { }

      template <typename T2>
      inline
      fvar<T>&
      operator+=(const fvar<T2>& x2) {
        val_ += x2.val_;
        d_ += x2.d_;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator+=(const T2& x2) {
        val_ += x2;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator-=(const fvar<T2>& x2) {
        val_ -= x2.val_;
        d_ -= x2.d_;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator-=(const T2& x2) {
        val_ -= x2;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator*=(const fvar<T2>& x2) {
        d_ = d_ * x2.val_ + val_ * x2.d_;
        val_ *= x2.val_;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator*=(const T2& x2) {
        val_ *= x2;
        return *this;
      }

      // SPEEDUP: specialize for T2 == var with d_ function

      template <typename T2>
      inline
      fvar<T>&
      operator/=(const fvar<T2>& x2) {
        d_ = (d_ * x2.val_ - val_ * x2.d_) / ( x2.val_ * x2.val_);
        val_ /= x2.val_;
        return *this;
      }

      template <typename T2>
      inline
      fvar<T>&
      operator/=(const T2& x2) {
        val_ /= x2;
        return *this;
      }

      inline
      fvar<T>&
      operator++() {
        ++val_;
        return *this;
      }
      inline
      fvar<T>
      operator++(int /*dummy*/) {
        fvar<T> result(val_,d_);
        ++val_;
        return result;
      }

      inline
      fvar<T>&
      operator--() {
        --val_;
        return *this;
      }
      inline
      fvar<T>
      operator--(int /*dummy*/) {
        fvar<T> result(val_,d_);
        --val_;
        return result;
      }

      
    };

//binary infix operators and unary prefix operators
    template <typename T>
    inline 
    fvar<T>
    operator-(const fvar<T>& x) {
      return fvar<T>(-x.val_, -x.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator+(const fvar<T1>& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ + x2.val_, 
                                                  x1.d_ + x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator+(const T1& x1,
              const fvar<T2>& x2) {
      return 
        fvar<typename 
             stan::return_type<T1,T2>::type>(x1 + x2.val_, 
                                             x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator+(const fvar<T1>& x1, 
              const T2& x2) {
      return 
        fvar<typename 
             stan::return_type<T1,T2>::type>(x1.val_ + x2, 
                                             x1.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const fvar<T1>& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ - x2.val_, 
                                                  x1.d_ - x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const T1& x1,
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1 - x2.val_, 
                                                  -x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const fvar<T1>& x1, 
              const T2& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ - x2, 
                                                  x1.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator*(const fvar<T1>& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ * x2.val_, 
                                                  x1.d_ * x2.val_ 
                                                  + x1.val_ * x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator*(const T1& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1 * x2.val_, 
                                                  x1 * x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator*(const fvar<T1>& x1, 
              const T2& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ * x2,
                                                  x1.d_ * x2);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator/(const fvar<T1>& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan
                  ::return_type<T1,T2>::type>(x1.val_ / x2.val_, 
                                              ( x1.d_ * x2.val_ 
                                                - x1.val_ * x2.d_ )
                                              / (x2.val_ * x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator/(const fvar<T1>& x1, 
              T2& x2) {
      return fvar<typename stan::return_type<T1,T2>::type>(x1.val_ / x2,
                                                           x1.d_ / x2);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator/(const T1& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan
                  ::return_type<T1,T2>::type>(x1 / x2.val_, 
                                              - x1 * x2.d_ 
                                              / (x2.val_ * x2.val_));
    }

//power and log functions
    template <typename T>
    inline
    fvar<T>
    sqrt(const fvar<T>& x) {
      using std::sqrt;
      return fvar<T>(sqrt(x.val_),
                     x.d_ / (2 * sqrt(x.val_)));
    }

    template <typename T>
    inline
    fvar<T>
    cbrt(const fvar<T>& x) {
      using boost::math::cbrt;
      return fvar<T>(cbrt(x.val_),
                     x.d_ / ( cbrt(x.val_) * cbrt(x.val_) * 3.0));
    }

    template <typename T>
    inline
    fvar<T>
    square(const fvar<T>& x) {
      using stan::math::square;
      return fvar<T>(square(x.val_),
                     x.d_ * 2 * x.val_);
    }

    template <typename T>
    inline
    fvar<T>
    exp(const fvar<T>& x) {
      using std::exp;
      return fvar<T>(exp(x.val_),
                     x.d_ * exp(x.val_));
    }

    template <typename T>
    inline
    fvar<T>
    exp2(const fvar<T>& x) {
      using stan::math::exp2;
      using std::log;
      return fvar<T>(exp2(x.val_),
                     x.d_ * exp2(x.val_) * log(2));
    }

    template <typename T>
    inline
    fvar<T>
    log(const fvar<T>& x) {
      using std::log;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
	  return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
          return fvar<T>(log(x.val_),
                     x.d_ / x.val_);
    }

    template <typename T>
    inline
    fvar<T>
    log2(const fvar<T>& x) {
      using std::log;
      using stan::math::log2;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
	  return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
          return fvar<T>(log2(x.val_),
			 x.d_ / (x.val_ * log(2)));
    }

    template <typename T>
    inline
    fvar<T>
    log10(const fvar<T>& x) {
      using std::log;
      using std::log10;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
	  return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
          return fvar<T>(log10(x.val_),
			 x.d_ / (x.val_ * log(10)));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    pow(const fvar<T1>& x1, 
              const T2& x2) {
      using std::pow;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>( pow(x1.val_, x2),
					   x1.d_ * x2 * pow(x1.val_, x2 - 1));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    pow(const T1& x1, 
	const fvar<T2>& x2) {
      using std::pow;
      using std::log;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>( pow(x1, x2.val_),
				      	  x2.d_ * log(x1) * pow(x1, x2.val_));
    }

//trig functions
    template <typename T>
    inline
    fvar<T>
    sin(const fvar<T>& x) {
      using std::sin;
      using std::cos;
      return fvar<T>(sin(x.val_),
                     x.d_ * cos(x.val_));
    }

    template <typename T>
    inline
    fvar<T>
    cos(const fvar<T>& x) {
      using std::sin;
      using std::cos;
      return fvar<T>(cos(x.val_),
                     x.d_ * -sin(x.val_));
    }

    template <typename T>
    inline
    fvar<T>
    tan(const fvar<T>& x) {
      using std::cos;
      using std::tan;
      return fvar<T>(tan(x.val_),
                     x.d_ / (cos(x.val_) * cos(x.val_)));
    }

    template <typename T>
    inline
    fvar<T>
    asin(const fvar<T>& x) {
      using std::asin;
      using std::sqrt;
      return fvar<T>(asin(x.val_),
                     x.d_ / sqrt(1 - x.val_ * x.val_));
    }

    template <typename T>
    inline
    fvar<T>
    acos(const fvar<T>& x) {
      using std::acos;
      using std::sqrt;
      return fvar<T>(acos(x.val_),
                     x.d_ / -sqrt(1 - x.val_ * x.val_));
    }

    template <typename T>
    inline
    fvar<T>
    atan(const fvar<T>& x) {
      using std::atan;
      return fvar<T>(atan(x.val_),
                     x.d_ / (1 + x.val_ * x.val_));
    }
  }
}
#endif
