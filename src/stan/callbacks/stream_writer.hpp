#ifndef STAN_CALLBACKS_STREAM_WRITER_HPP
#define STAN_CALLBACKS_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace callbacks {

    /**
     * <code>stream_writer</code> is an implementation
     * of <code>writer</code> that writes to a stream.
     */
    class stream_writer : public writer {
    public:
      /**
       * Constructs a stream writer with an output stream
       * and an optional prefix for comments.
       *
       * @param[in, out] output stream to write
       * @param[in] comment_prefix string to stream before
       *   each comment line. Default is "".
       */
      stream_writer(std::ostream& output,
                    const std::string& comment_prefix = ""):
        output_(output), comment_prefix_(comment_prefix) {}

      /**
       * Writes a set of names on a single line in csv format followed
       * by a newline.
       *
       * Note: the names are not escaped.
       *
       * @param[in] names Names in a std::vector
       */
      void operator()(const std::vector<std::string>& names) {
        if (names.empty()) return;

        std::vector<std::string>::const_iterator last = names.end();
        --last;

        for (std::vector<std::string>::const_iterator it = names.begin();
             it != last; ++it)
          output_ << *it << ",";
        output_ << names.back() << std::endl;
      }

      /**
       * Writes a set of values in csv format followed by a newline.
       *
       * Note: the precision of the output is determined by the settings
       *  of the stream on construction.
       *
       * @param[in] state Values in a std::vector
       */
      void operator()(const std::vector<double>& state) {
        if (state.empty()) return;

        std::vector<double>::const_iterator last = state.end();
        --last;

        for (std::vector<double>::const_iterator it = state.begin();
             it != last; ++it)
          output_ << *it << ",";
        output_ << state.back() << std::endl;
      }

      /**
       * Writes the comment_prefix to the stream followed by a newline.
       */
      void operator()() {
        output_ << comment_prefix_ << std::endl;
      }

      /**
       * Writes the comment_prefix then the message followed by a newline.
       *
       * @param[in] message A string
       */
      void operator()(const std::string& message) {
        output_ << comment_prefix_ << message << std::endl;
      }

      /**
       * Virtual destructor
       */
      virtual ~stream_writer() {}

    private:
      /**
       * Output stream
       */
      std::ostream& output_;
      /**
       * Comment prefix to use when printing comments: strings and blank lines
       */
      std::string comment_prefix_;
    };

  }
}
#endif
