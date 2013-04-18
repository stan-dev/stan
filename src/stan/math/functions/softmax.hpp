#ifndef __STAN__MATH__FUNCTIONS__SOFTMAX_HPP__
#define __STAN__MATH__FUNCTIONS__SOFTMAX_HPP__

#include <boost/math/tools/promotion.hpp>
#include <stdexcept>
#include <boost/throw_exception.hpp>

namespace stan {
  namespace math {

    // hide helper for now; could use Eigen here
    namespace {
      template <typename Vector, typename Scalar>
      Scalar maximum(const Vector& x) {
        if(x.size() == 0)
          BOOST_THROW_EXCEPTION(std::invalid_argument ("x must have at least one element"));
        Scalar max_x(x[0]);
        for (typename Vector::size_type i = 1; i < x.size(); ++i)
          if (x[i] < max_x)
            max_x = x[i];
        return max_x;
      }
    }

    /**
     * Write the values of the softmax transformed first argument
     * into the second argument.  Values in the first argument
     * are unbounded and values in the output form a simplex.
     *
     * The softmax transformed generalizes the inverse logistic
     * function by transforming a vector \f$x\f$ of length \f$K\f$ as
     *
     * \f$\mbox{softmax}(x)[i] = \frac{\exp(x[i])}{\sum_{k=1}^{K} \exp(x[k])}\f$.
     *
     * By construction, the result is a simplex, which means the values
     * are all non-negative and sum to 1.0, 
     * 
     * \f$ \sum_{k=1}^{K} \mbox{softmax}(x)[k] = 1\f$.
     *
     * The type <code>Vector</code> is for a vector with values of
     * type <code>Scalar</code>.  Vectors <code>x</code> must support
     * <code>x.size()</code> and return assignable references of type
     * <code>Scalar</code> through <code>x[int]</code>.  Variables
     * <code>a</code> of type <code>Scalar</code> need to support
     * division (in context <code>x[i] /= a</code>) and exponentiation
     * (<code>exp(a)</code>).  Conforming examples include
     *
     * <code>Vector = std::vector&lt;double&gt;</code> and
     * <code>Scalar = double</code>, 
     *
     * <code>Vector = std::vector&lt;stan::agrad::var&gt;</code> and
     * <code>Scalar = stan::agrad::var</code>, 
     * 
     * <code>Vector = Eigen::Matrix<double,Eigen::Dynamic,1>&lt;double&gt;</code> and
     * <code>Scalar = double</code>,
     *
     * and so on.
     *
     * The function <code>stan::math::inverse_softmax</code> provides an
     * inverse of this operation up to an additive constant.  Specifically,
     * if <code>x</code> is a simplex argument, then
     *
     * <code>softmax(inv_softmax(x)) = x</code>.
     *
     * If <code>y</code> is an arbitrary vector, then we only have
     * identification up to an additive constant <code>c</code>, as in
     *
     * <code>inv_softmax(softmax(y)) = y + c * I</code>.
     *
     * @param x Input vector of unbounded parameters.
     * @param simplex Output vector of simplex values.
     * @throw std::invalid_argument if size of the input and output vectors differ.
     */
    template <typename Vector, typename Scalar>
    void softmax(const Vector& x, Vector& simplex) {
      using std::exp;
      if (x.size() != simplex.size()) 
        BOOST_THROW_EXCEPTION(std::invalid_argument ("x.size() != simplex.size()"));
      Scalar sum(0.0); 
      Scalar max_x = maximum<Vector,Scalar>(x);
      for (typename Vector::size_type i = 0; i < x.size(); ++i) {
        simplex[i] = exp(x[i]-max_x);
        sum += simplex[i];
      }
      for (typename Vector::size_type i = 0; i < x.size(); ++i)
        simplex[i] /= sum;
    }

  }
}

#endif
