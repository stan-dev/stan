#ifndef STAN_INTERFACE_CALLBACKS_WRITER_SUM_VALUES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_SUM_VALUES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      /**
       * Writer that only keeps the sum of the state values.
       */
      class sum_values: public base_writer {
      public:
        explicit sum_values(const size_t N)
          : N_(N), m_(0), skip_(0), sum_(N_, 0.0) { }

        sum_values(const size_t N, const size_t skip)
          : N_(N), m_(0), skip_(skip), sum_(N_, 0.0) { }

        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        double value) { }

        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        const std::string& value) { }

        /**
         * Writes a key and an array of values.
         * This implementation does nothing.
         *
         * @param key The key
         * @param values The values
         * @param n_values The number of values
         */
        void operator()(const std::string& key,
                        const double* values,
                        int n_values)  { }

        /**
         * Writes a key and a 2-d array of values.
         * This implementation does nothing.
         *
         * @param key The key
         * @param values The values stored in row-major order
         * @param n_rows The number of rows
         * @param n_cols The number of columns
         */
        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) { }

        /**
         * Writes a vector of names.
         * This implementation does nothing.
         *
         * @param names Names to write
         */
        void operator()(const std::vector<std::string>& names) { }

        /**
         * Writes a message.
         * This implementation does nothing.
         *
         * @param message Message to write
         */
        void operator()(const std::string& message) { }

        /**
         * Writes nothing.
         * This implementation does nothing.
         *
         * An implementation may choose to treat this as a flush.
         */
        void operator()() { }

        /**
         * Writes a vector of states.
         *
         * @param state State to write
         */
        void operator()(const std::vector<double>& state) {
          if (N_ != state.size())
            throw std::length_error("vector provided does not "
                                    "match the parameter length");
          if (m_ >= skip_) {
            for (size_t n = 0; n < N_; n++)
              sum_[n] += state[n];
          }
          m_++;
        }

        /**
         * Returns the sum of the states.
         *
         * @return a constant vector with the sums of each of the
         * states
         */
        const std::vector<double>& sum() const {
          return sum_;
        }

        /**
         * Returns the number of times this has been called.
         *
         * @return the number of times this writer has been called
         */
        const size_t called() const {
          return m_;
        }

        /**
         * Returns the number of times the writer has actually recorded
         * values.
         *
         * @return the number of times the writer has recorded values. This
         * is the difference between the number of times called and number to
         * skip.
         */
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
