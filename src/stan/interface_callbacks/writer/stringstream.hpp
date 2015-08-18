#ifndef STAN_INTERFACE_CALLBACKS_WRITER_STRINGSTREAM_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_STRINGSTREAM_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <sstream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * An implementation of base_writer using stringstream
       *
       */
      class stringstream: public base_writer {
      private:
        std::stringstream& stream_;
      public:
        /**
         * Constructor.
         * 
         * @param stream A valid stringstream
         */
        explicit stringstream(std::stringstream& stream)
          : stream_(stream) {
        }

        /**
         * Writes a message.
         *
         * @param message Message to write
         */
        void operator()(const std::string& message) {
          stream_ << message << std::endl;
        }

        /**
         * Writes nothing.
         *
         * An implementation may choose to treat this as a flush.
         */
        void operator()() {
          stream_ << std::endl;
        }
      };

    }
  }
}

#endif
