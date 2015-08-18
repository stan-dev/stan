#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <string>
#include <sstream>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * Base writer class.
       *
       * This is a abstract base class. The only functions
       * an implementing class needs to implement are
       * - void operator()(const std::string& message)
       * - void operator()()
       */
      class base_writer {
      public:
        /**
         * Virtual destructor.
         */
        virtual ~base_writer() {}

        /**
         * Writes a key, value pair.
         *
         * @param key The key
         * @param value The value
         */
        virtual void operator()(const std::string& key,
                                double value) {
          std::stringstream ss;
          ss << key << " = " << value;
          this->operator()(ss.str());
        }

        /**
         * Writes a key, value pair.
         *
         * @param key The key
         * @param value The value
         */
        virtual void operator()(const std::string& key,
                                const std::string& value) {
          this->operator()(key + " = " + value);
        }

        /**
         * Writes a key and an array of values.
         *
         * @param key The key
         * @param values The values
         * @param n_values The number of values
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_values)  {
          std::stringstream ss;
          ss << key << " = ";
          if (n_values > 0)
            ss << values[0];
          for (int n = 1; n < n_values; n++)
            ss << "," << values[n];
          this->operator()(ss.str());
        }

        /**
         * Writes a key and a 2-d array of values.
         *
         * @param key The key
         * @param values The values stored in row-major order
         * @param n_rows The number of rows
         * @param n_cols The number of columns
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) {
          std::stringstream ss;
          ss << key << " = [";
          if (n_rows > 0 && n_cols > 0) {
            for (int i = 0; i < n_rows; i++) {
              ss << values[i * n_cols];
              for (int j = 1; j < n_cols; j++)
                ss << "," << values[i * n_cols + j];
              if (i != n_rows - 1)
                ss << std::endl;
            }
          }
          ss << "]";
          this->operator()(ss.str());
        }

        /**
         * Writes a vector of names.
         *
         * @param names Names to write
         */
        virtual void operator()(const std::vector<std::string>& names) {
          std::stringstream ss;
          if (names.size() > 0)
            ss << names[0];
          for (size_t n = 1; n < names.size(); n++)
            ss << "," << names[n];
          this->operator()(ss.str());
        }

        /**
         * Writes a vector of states.
         *
         * @param state State to write
         */
        virtual void operator()(const std::vector<double>& state) {
          std::stringstream ss;
          if (state.size() > 0)
            ss << state[0];
          for (std::vector<double>::const_iterator it = state.begin()+1;
               it != state.end(); it++)
            ss << "," << *it;
          this->operator()(ss.str());
        }

        /**
         * Writes a message.
         *
         * @param message Message to write
         */
        virtual void operator()(const std::string& message) = 0;

        /**
         * Writes nothing.
         *
         * An implementation may choose to treat this as a flush.
         */
        virtual void operator()() = 0;
      };

    }
  }
}

#endif
