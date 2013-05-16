#ifndef __STAN__AGRAD__FWD__FVAR__HPP__
#define __STAN__AGRAD__FWD__FVAR__HPP__

namespace stan {

  namespace agrad {

    template <typename T>
    struct fvar {

      T val_;  // value
      T d_;    // tangent (aka derivative)

      T val() { return val_; }
      T tangent() { return d_; }

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
  }
}
#endif
