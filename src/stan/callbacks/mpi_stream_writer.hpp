#ifndef STAN_CALLBACKS_MPI_STREAM_WRITER_HPP
#define STAN_CALLBACKS_MPI_STREAM_WRITER_HPP

#ifdef MPI_ADAPTED_WARMUP

#include <stan/callbacks/writer.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace callbacks {
    /**
     * <code>mpi_stream_writer</code> is an implementation
     * of <code>writer</code> that writes to a stream.
     */
    class mpi_stream_writer : public writer {
    public:
      /**
       * Constructs a stream writer with an output stream
       * and an optional prefix for comments.
       *
       * @param[in, out] output stream to write
       * @param[in] comment_prefix string to stream before
       *   each comment line. Default is "".
       */
      mpi_stream_writer(int num_chains, std::ostream& output,
                        const std::string& comment_prefix = "")
        : num_chains_(num_chains), output_(output),
          comment_prefix_(comment_prefix)
      {}

      /**
       * Virtual destructor
       */
      virtual ~mpi_stream_writer() {}

      /**
       * Set new value for @c num_chains_.
       *
       * @param[in] n new value of @c num_chains_
       */
      void set_num_chains(int n) {
        num_chains_ = n;
      }

      /**
       * Writes a set of names on a single line in csv format followed
       * by a newline.
       *
       * Note: the names are not escaped.
       *
       * @param[in] names Names in a std::vector
       */
      void operator()(const std::vector<std::string>& names) {
        write_vector(names);
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
        write_vector(state);
      }

      /**
       * Writes the comment_prefix to the stream followed by a newline.
       */
      void operator()() {
        if (stan::math::mpi::Session::is_in_inter_chain_comm(num_chains_)) {
          output_ << comment_prefix_ << std::endl;
        }
      }

      /**
       * Writes the comment_prefix then the message followed by a newline.
       *
       * @param[in] message A string
       */
      void operator()(const std::string& message) {
        if (stan::math::mpi::Session::is_in_inter_chain_comm(num_chains_)) {
          output_ << comment_prefix_ << message << std::endl;
        }
      }

    private:

      /**
       * nb. of chains that have its own output stream
       */
      int num_chains_;

      /**
       * Output stream
       */
      std::ostream& output_;

      /**
       * Comment prefix to use when printing comments: strings and blank lines
       */
      std::string comment_prefix_;

      /**
       * Writes a set of values in csv format followed by a newline.
       *
       * Note: the precision of the output is determined by the settings
       *  of the stream on construction.
       *
       * @param[in] v Values in a std::vector
       */
      template <class T>
      void write_vector(const std::vector<T>& v) {
        if (stan::math::mpi::Session::is_in_inter_chain_comm(num_chains_)) {
          if (v.empty()) return;

          typename std::vector<T>::const_iterator last = v.end();
          --last;

          for (typename std::vector<T>::const_iterator it = v.begin();
               it != last; ++it)
            output_ << *it << ",";
          output_ << v.back() << std::endl;
        }
      }
    };

  }
}

#endif

#endif
