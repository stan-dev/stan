#ifndef __STAN__MATH__MATRIX__ACCUMULATOR_HPP__
#define __STAN__MATH__MATRIX__ACCUMULATOR_HPP__

#include <vector>
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

      ~accumulator() { }
      
      /**
       * Add the specified value to the buffer.
       *
       * @param x Value to add.
       */
      void add(const T& x) {
        buf_.push_back(x);
      }
      
      /**
       * Return the sum of the accumulated values.
       *
       * @return Sum of accumulated values.
       */
      T sum() {
        using math::sum;
        return sum(buf_);
      }

    };




  }
}

#endif
