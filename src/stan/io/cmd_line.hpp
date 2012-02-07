#ifndef __STAN__IO__CMD_LINE_HPP__
#define __STAN__IO__CMD_LINE_HPP__

#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <ostream>

namespace stan {

  namespace io {
    
    /**
     * The <code>cmd_line</code> class parses and stores command-line
     * arguments.
     *
     * <p>Command-line arguments are organized into four types.
     *
     * <p><b>Command</b>: The first argument (at index 0) is just the
     * command itself.  There method <code>command()</code> retrieves
     * the command.
     *
     * <p><b>Key/Value</b>: The second type of argument is a key-value pair, 
     * which must be in the form <code>--key=val</code>.  Two hyphens
     * are used to separate arguments from negated numbers.  The method
     * <code>has_key(const std::string&)</code> indicates if there is a key
     * and <code>val(const std::string&,T&)</code> writes its value into
     * a reference (whose type is templated; any type understand by the
     * output operator <code>&gt;&gt;</code> is acceptable.
     *
     * <p><b>Flag</b>: Flags are specified as <code>--flag</code>.  The
     * method <code>has_flag(const std::string&)</code> tests if a flag
     * is present.
     *
     * <p><b>Bare Argument</b>: Bare arguments are any arguments that
     * are not prefixed with two hyphens (<code>--</code>).  The
     * method <code>bare_size()</code> returns the number of bare
     * arguments and they are retrieved with the generic method
     * <code>bare(const std::string&,T&)</code>.
     */
    class cmd_line {
    private:
      std::string cmd_;
      std::map<std::string,std::string> key_val_;
      std::set<std::string> flag_;
      std::vector<std::string> bare_;
      void parse_arg(const std::string& s) {
        if (s.size() < 2
            || s[0] != '-'
            || s[1] != '-') {
          bare_.push_back(s);
          return;
        }
        for (size_t i = 2; i < s.size(); ++i) {
          if (s[i] == '=') {
            key_val_[s.substr(2,i - 2)] = s.substr(i + 1,s.size() - i - 1);
            return;
          } 
        }
        flag_.insert(s.substr(2,s.size()));
      }
    public:
      /**
       * Construct a command-line argument object from the specified
       * command-line arguments.
       *
       * @param argc Number of arguments.
       * @param argv Argument strings.
       */
      cmd_line(int argc, const char* argv[])
        : cmd_(argv[0]) {
        for (int i = 1; i < argc; ++i) 
          parse_arg(argv[i]);
      }

      /**
       * Returns the name of the command itself.  The
       * command is always supplied as the first argument
       * (at index 0).
       *
       * @return Name of command.
       */
      std::string command() {
        return cmd_;
      }
      
      /**
       * Return <code>true</code> if the specified key is defined.
       *
       * @param key Key to test.
       * @return <code>true</code> if it has a value.
       */
      bool has_key(const std::string& key) const {
        return key_val_.find(key) != key_val_.end();
      }

      /**
       * If the specified key is defined, write the value of the key
       * into the specified reference and return <code>true</code>,
       * otherwise do not modify the reference and return
       * <code>false</code>.
       *
       * <p>The conversions defined by <code>std::ostream</code> 
       * are used to convert the base string value to the specified
       * type.  Thus this method will work as long as <code>operator>>()</code>
       * is defined for the specified type.
       *
       * @param key Key whose value is returned.
       * @param x Reference to value.
       * @return Value for key, or empty string if it's not defined.
       * @tparam Type of value.
       */
      template <typename T>
      bool val(const std::string& key, T& x) const {
        if (!has_key(key))
          return false;
        std::stringstream s(key_val_.find(key)->second);
        return s >> x;
      }

      /**
       * Return <code>true</code> if the specified flag is defined.
       *
       * @param flag Flag to test.
       * @return <code>true</code> if flag is defined.
       */
      bool has_flag(const std::string& flag) const {
        return flag_.find(flag) != flag_.end();
      }

      /**
       * Return the number of bare arguments.
       *
       * @return Number of bare arguments.
       */
      size_t bare_size() const {
        return bare_.size();
      }

      /**
       * If the specified index is valid for bare arguments,
       * write the bare argument at the specified index into 
       * the specified reference, and otherwise return false
       * without modifying the reference.
       *
       * @param n Bare argument position.
       * @param x Reference to result.
       * @return <code>true</code> if there were enough bare arguments.
       * @tparam T Type of value returned.
       */
      template <typename T>
      bool bare(size_t n, T& x) const {
        if (n >= bare_.size())
          return false;
        std::stringstream s(bare_[n]);
        s >> x;
        return true;
      }

      /**
       * Print a human readable parsed form of the command-line
       * arguments to the specified output stream.
       *
       * @param out Output stream.
       */
      void print(std::ostream& out) const {
        out << "COMMAND=" << cmd_ << '\n';
        size_t flag_count = 0;
        for (std::set<std::string>::const_iterator it = flag_.begin();
             it != flag_.end();
             ++it) {
          out << "FLAG " << flag_count << "=" << (*it) << '\n';
          ++flag_count;
        }
        size_t key_val_count = 0;
        for (std::map<std::string,std::string>::const_iterator it = key_val_.begin();
             it != key_val_.end();
             ++it) {
          out << "KEY " << key_val_count << "=" << (*it).first;
          out << " VAL " << key_val_count << "=" << (*it).second << '\n';
          ++key_val_count;
        }
        size_t bare_count = 0;
        for (size_t i = 0; i < bare_.size(); ++i) {
          out << "BARE ARG " << bare_count << "=" << bare_[i] << '\n';
          ++bare_count;
        }
      }

    };


  }

}

#endif
