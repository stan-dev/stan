#ifndef STAN_IO_PROGRAM_READER_PROGRAM_READER_HPP
#define STAN_IO_PROGRAM_READER_PROGRAM_READER_HPP

#include <stan/io/read_line.hpp>
#include <stan/io/starts_with.hpp>
#include <cstdio>
#include <istream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <iostream>  // TODO(carpenter): remove this line

namespace stan {
  namespace io {

    /**
     * Structure to hold preprocessing events, which hold (a) line
     * number in concatenated program after includes, (b) line number
     * in the stream from which the text is read, (c) a string-based
     * action, and (d) a path to the current file.
     */
    struct preproc_event {
      int concat_line_num_;
      int line_num_;
      std::string action_;
      std::string path_;
      preproc_event(int concat_line_num, int line_num,
                    const std::string& action, const std::string& path)
        : concat_line_num_(concat_line_num), line_num_(line_num),
          action_(action), path_(path) { }
    };

    /**
     * A <code>program_reader</code> reads a Stan program and unpacks
     * the include statements relative to a search path in such a way
     * that error messages can reproduce the include path.
     */
    class program_reader {
    public:

      /**
       * Construct a program reader from the specified stream derived
       * from the specified name or path, and a sequence of paths to
       * search for include files.  The paths should be directories.
       *
       * <p>It is up to the caller that created the input stream to
       * close it.
       *
       * @param[in] in stream from which to read
       * @param[in] name name or path attached to stream
       * @param[in] search_path ordered sequence of directory names to
       * search for included files
       */
      program_reader(std::istream& in, const std::string& name,
                     const std::vector<std::string>& search_path) {
        int concat_line_num = 0;
        read(in, name, search_path, concat_line_num);
      }

      /**
       * Return a stream from which to read the concatenated program.
       * Modifying the stream will modify the underlying class.
       *
       * @return stream for program
       */
      std::stringstream& program_stream() {
        return program_;
      }

      /**
       * Return the include message for the target line number.  This
       * will take the form
       *
       * <pre>
       * in file '<file>' at line <num>
       * included from file '<file>' at line <num>
       * ...
       * included from file '<file> at line <num>
       * </pre>
       *
       * @param target_line_num line number in concatenated program
       * @return include trace for the line number
       */
      std::string include_trace(int target_line_num) const {
        const dumps_t x = include_stack(target_line_num);
        std::stringstream ss;
        ss << "in file '" << x[x.size() - 1].first
           << "' at line " << x[x.size() - 1].second
           << std::endl;
        for (size_t i = x.size() - 1; i-- > 0; )
          ss << "included from file '" << x[i].first
             << "' at line " << x[i].second
             << std::endl;
        return ss.str();
      }

      // temp to print the history out for debugging
      void print_history(std::ostream& out) {
        for (size_t i = 0; i < history_.size(); ++i)
          out << i << ". (" << history_[i].concat_line_num_
              << ", " << history_[i].line_num_
              << ", " << history_[i].action_
              << ", " << history_[i].path_ << ")"
              << std::endl;
      }


    private:
      std::stringstream program_;
      std::vector<preproc_event> history_;

      /**
       * A path/position pair.
       */
      typedef std::pair<std::string, int> dump_t;

      /**
       * Sequence of path/position pairs.
       */
      typedef std::vector<dump_t> dumps_t;

      /**
       * Return the sequence of path/line number pairs identifying
       * where the target line number came from.
       *
       * @param[in] target_line_num line number in concatenated
       * program file
       * @return sequence of files and positions for includes
       */
      dumps_t include_stack(int target_line_num) const {
        dumps_t result;
        std::string local_file;  // file currently in
        int local_line_num;      // where local line num started
        int global_line_num;     // where global line num started
        for (size_t hist_pos = 0; hist_pos < history_.size(); ++hist_pos) {
          if (history_[hist_pos].action_ == "start"
              || history_[hist_pos].action_ == "restart" ) {
            local_file = history_[hist_pos].path_;
            local_line_num = history_[hist_pos].line_num_;
            global_line_num = history_[hist_pos].concat_line_num_;
          } else if (history_[hist_pos].action_ == "end") {
            if (target_line_num < history_[hist_pos].concat_line_num_) {
              int n = target_line_num - global_line_num + local_line_num;
              result.push_back(dump_t(local_file, n));
              break;
            }
            result.pop_back();
          } else if (history_[hist_pos].action_ == "include") {
            result.push_back(dump_t(local_file,
                                    history_[hist_pos].line_num_));
          }
        }
        return result;
      }

      /**
       * Returns the characters following <code>#include</code> on
       * the line, trimming whitespace characters.  Assumes that
       * <code>#include</code>" is line initial.
       *
       * @param line line of text beginning with <code>#include</code>
       * @return text after <code>#include</code> with whitespace
       * trimmed
       */
      static std::string include_path(const std::string& line) {
        int start = std::string("#include").size();
        while (line[start] == ' ') ++start;
        int end = line.size() - 1;
        while (line[end] == ' ') --end;
        return line.substr(start, end - start);
      }

      // TODO(carpenter): need try/catch to guarantee closure of stream
      /**
       * Read the rest of a program from the specified input stream in
       * the specified path, with the specified search path for
       * include files, and incrementing the specified concatenated
       * line number.  This method is called recursively for included
       * files.
       *
       * @param[in] in stream from which to read
       * @param[in] path name of stream
       * @param[in] search_path sequence of path names to search for
       * include files
       * @param[in,out] concat_line_num position in concatenated file
       * to be updated
       * @throw std::runtime_error if an included file cannot be found
       */
      void read(std::istream& in, const std::string& path,
                const std::vector<std::string>& search_path,
                int& concat_line_num) {
        history_.push_back(preproc_event(concat_line_num, 0, "start", path));
        for (int line_num = 1; ; ++line_num) {
          std::cout << path << " (" << line_num << ")" << std::endl;
          std::string line = read_line(in);
          if (line.empty()) {
            // ends initial out of loop start event
            history_.push_back(preproc_event(concat_line_num, line_num,
                                             "end", path));
            break;
          } else if (starts_with("#include ", line)) {
            std::string incl_path = include_path(line);
            history_.push_back(preproc_event(concat_line_num, line_num,
                                             "include", incl_path));
            bool found_path = false;
            for (size_t i = 0; i < search_path.size(); ++i) {
              std::string f = search_path[i] + incl_path;
              std::ifstream include_in(f.c_str());
              if (!include_in.good()) {
                include_in.close();
                continue;
              }
              try {
                read(include_in, incl_path, search_path, concat_line_num);
              } catch (...) {
                include_in.close();
                throw;
              }
              include_in.close();
              history_.push_back(preproc_event(concat_line_num, line_num,
                                               "restart", path));
              found_path = true;
              break;
            }
            if (!found_path)
              throw std::runtime_error("could not find include file");
          } else {
            ++concat_line_num;
            program_ << line;
          }
        }
      }

    };

  }
}
#endif

