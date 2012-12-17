#ifndef __STAN__AGRAD__FVAR_HPP__
#define __STAN__AGRAD__FVAR_HPP__

#include <stan/meta/traits.hpp>

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
      operator-=(const fvar<T2>& x2) {
        val_ -= x2.val_;
        d_ -= x2.d_;
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

      // SPEEDUP: specialize for T2 == var with d_ function

      template <typename T2>
      inline
      fvar<T>&
      operator/=(const fvar<T2>& x2) {
        d_ = (d_ * x2.val_ - val_ * x2.d_) / ( x2.v_ * x2.v_);
        val_ /= x2.val_;
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
      return 
        fvar<typename 
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
                                              - x1.val_ * x2.d_ 
                                              / (x2.val_ * x2.val_));
    }


  }
}


#endif
