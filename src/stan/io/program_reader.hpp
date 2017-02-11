#ifndef STAN_IO_PROGRAM_READER_HPP
#define STAN_IO_PROGRAM_READER_HPP

#include <cstdio>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <iostream>
#include <stdexcept>

namespace stan {
  namespace io {

    bool starts_with(const std::string& prefix,
                     const std::string& s) {
      return s.size() >= prefix.size()
        && s.substr(0, prefix.size()) == prefix;
    }

    std::string read_line(std::istream& in) {
      std::stringstream ss;
      while (true) {
        int c = in.get();
        if (c == std::char_traits<char>::eof()) return ss.str();
        ss << static_cast<char>(c);
        if (c == '\n') return ss.str();
      }
    }

    bool is_include(const std::string& line) {
      return starts_with("#include", line);
    }

    // assumes is_include(line) is true
    std::string include_path(const std::string& line) {
      int start = std::string("#include").size();
      if (line[start] != ' ') {
        throw std::runtime_error("expect space after #include");
      }
      while (line[start] == ' ') ++start;
      int end = line.size() - 1;
      while (line[end] == ' ') --end;
      return line.substr(start, end - start);
    }

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

    class program_reader {
    public:
      program_reader(std::istream& in, const std::string& path,
                     const std::vector<std::string>& search_path) {
        read(in, path, search_path);
      }

      std::stringstream& program_stream() {
        return program_;
      }

      void print_history(std::ostream& out) {
        for (size_t i = 0; i < history_.size(); ++i)
          out << i << ". (" << history_[i].concat_line_num_
              << ", " << history_[i].line_num_
              << ", " << history_[i].action_
              << ", " << history_[i].path_ << ")"
              << std::endl;
      }

      typedef std::pair<std::string, int> dump_t;
      typedef std::vector<dump_t> dumps_t;

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
            result.push_back(dump_t(local_file, history_[hist_pos].line_num_));
          }
        }
        return result;
      }

      static std::string render(const dumps_t & x) {
        for (size_t i = 0; i < x.size(); ++i)
          std::cout << "render(" << i << ") = "
                    << x[i].first << ", " << x[i].second << std::endl;

        if (x.size() < 1)
          throw std::runtime_error("dump seq requires size > 0");
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

    private:
      std::stringstream program_;
      std::vector<preproc_event> history_;

      // TODO(carpenter): need try/catch to guarantee closure of stream

      void read(std::istream& in, const std::string& path,
                const std::vector<std::string>& search_path) {
        int concat_line_num = 0;
        return read(in, path, search_path, concat_line_num);
      }


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
          } else if (is_include(line)) {
            std::string incl_path = include_path(line);
            history_.push_back(preproc_event(concat_line_num, line_num,
                                             "include", incl_path));
            bool found_path = false;
            for (size_t i = 0; i < search_path.size(); ++i) {
              std::string f = search_path[i] + incl_path;
              std::ifstream include_in(f.c_str());
              if (!include_in.good()) continue;
              read(include_in, incl_path, search_path, concat_line_num);
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

