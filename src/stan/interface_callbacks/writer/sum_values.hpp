#ifndef STAN_INTERFACE_CALLBACKS_WRITER_SUM_VALUES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_SUM_VALUES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      class sum_values: public base_writer {
      public:
        explicit sum_values(const size_t N)
          : N_(N), m_(0), skip_(0), sum_(N_, 0.0) { }

        sum_values(const size_t N, const size_t skip)
          : N_(N), m_(0), skip_(skip), sum_(N_, 0.0) { }

        /**
         * Do nothing with std::string vector
         *
         * @tparam T type of element
         * @param x vector of type T
         */
        void operator()(const std::vector<std::string>& x) {
        }

        /**
         * Add values to cumulative sum
         *
         * @tparam T type of element
         * @param x vector of type T
         */
        template <class T>
        void operator()(const std::vector<T>& x) {
          if (N_ != x.size())
            throw std::length_error("vector provided does not "
                                    "match the parameter length");
          if (m_ >= skip_) {
            for (size_t n = 0; n < N_; n++)
              sum_[n] += x[n];
          }
          m_++;
        }


        /**
         * Do nothing with a string.
         *
         * @param x string to print with prefix in front
         */
        void operator()(const std::string& message) { }

        /**
         * Do nothing
         *
         */
        void operator()() { }

        const std::vector<double>& sum() const {
          return sum_;
        }

        const size_t called() const {
          return m_;
        }

        const size_t recorded() const {
          if (m_ >= skip_)
            return m_ - skip_;
          else
            return 0;
        }


      private:
        size_t N_;
        size_t m_;
        size_t skip_;
        std::vector<double> sum_;
      };

    }
  }
}

#endif
