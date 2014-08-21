#ifndef STAN__MATH__MATRIX__ACCUMULATOR_HPP
#define STAN__MATH__MATRIX__ACCUMULATOR_HPP

#include <vector>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_same.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/sum.hpp>

namespace stan {
  namespace math {

    /**
     * Class to accumulate values and eventually return their sum.  If
     * no values are ever added, the return value is 0.
     *
     * This class is useful for speeding up auto-diff of long sums
     * because it uses the <code>sum()</code> operation (either from
     * <code>stan::math</code> or one defined by argument-dependent lookup.
     *
     * @tparam T Type of scalar added
     */
    template <typename T>
    class accumulator {
    private:
      std::vector<T> buf_;

    public:
      /**
       * Construct an accumulator. 
       */
      accumulator()
        : buf_() {
      }

      /**
       * Destroy an accumulator.
       */
      ~accumulator() { }
      
      /**
       * Add the specified arithmetic type value to the buffer after
       * static casting it to the class type <code>T</code>.
       *
       * <p>See the Boost doc for <code>boost::is_arithmetic</code>
       * for information on what counts as an arithmetic type.
       *
       * @tparam S Type of argument
       * @param x Value to add
       */
      template <typename S>
      typename boost::enable_if<boost::is_arithmetic<S>, void>::type
      add(S x) {
        buf_.push_back(static_cast<T>(x));
      }

      /**
       * Add the specified non-arithmetic value to the buffer.
       *
       * <p>This function is disabled if the type <code>S</code> is
       * arithmetic or if it's not the same as <code>T</code>.
       *
       * <p>See the Boost doc for <code>boost::is_arithmetic</code>
       * for information on what counts as an arithmetic type.
       *
       * @tparam S Type of argument
       * @param x Value to add
       */
      template <typename S>
      typename boost::disable_if<boost::is_arithmetic<S>, 
                                 typename boost::enable_if<boost::is_same<S,T>,
                                                           void>::type >::type
      add(const S& x) {
        buf_.push_back(x);
      }

      /**
       * Add each entry in the specified matrix, vector, or row vector
       * of values to the buffer.
       *
       * @tparam S type of values in matrix
       * @tparam R number of rows in matrix
       * @tparam C number of columns in matrix
       * @param m Matrix of values to add
       */
      template <typename S, int R, int C>
      void add(const Eigen::Matrix<S,R,C>& m) {
        for (int i = 0; i < m.size(); ++i)
          add(m(i));
      }
      
      /**
       * Recursively add each entry in the specified standard vector
       * to the buffer.  This will allow vectors of primitives,
       * auto-diff variables to be added; if the vector entries
       * are collections, their elements are recursively added.
       *
       * @tparam S Type of value to recursively add.
       * @param xs Vector of entries to add
       */
      template <typename S>
      void add(const std::vector<S>& xs) {
        for (size_t i = 0; i < xs.size(); ++i)
          add(xs[i]);
      }
      
      /**
       * Return the sum of the accumulated values.
       *
       * @return Sum of accumulated values.
       */
      T sum() const {
        using math::sum;
        return sum(buf_);
      }

    };




  }
}

#endif
