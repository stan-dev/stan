#ifndef __STAN__IO__CMD_LINE_HPP__
#define __STAN__IO__CMD_LINE_HPP__

#include <map>
#include <set>
#include <vector>
#include <ostream>

namespace stan {

  namespace io {
    
    /**
     * The <code>cmd_line</code> class parses and stores command-line
     * arguments.
     */
    class cmd_line {
    private:
      std::string cmd_;
      std::map<std::string,std::string> key_val_;
      std::set<std::string> flag_;
      std::vector<std::string> bare_;
      void parse_arg(const std::string& s) {
	if (s.size() == 0
	    || s[0] != '-') {
	  bare_.push_back(s);
	  return;
	}
	unsigned int start_pos = 1U;
	if (s.size() > 1 && s[1] == '-')
	  start_pos = 2U;
	for (unsigned int i = start_pos; i < s.size(); ++i) {
	  if (s[i] == '=') {
	    key_val_[s.substr(start_pos,i - start_pos)] = s.substr(i + 1,s.size() - i - 1);
	    return;
	  } 
	}
	flag_.insert(s.substr(start_pos,s.size()));
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
       * Return <code>true</code> if the specified key is defined.
       *
       * @param key Key to test.
       * @return <code>true</code> if it has a value.
       */
      bool has_key(const std::string& key) const {
	return key_val_.find(key) != key_val_.end();
      }

      /**
       * Return the value for the specified key or the empty string
       * if it is not defined.
       *
       * @param key Key whose value is returned.
       * @return Value for key, or empty string if it's not defined.
       */
      std::string val(const std::string& key) const {
	if (has_key(key))
	  return key_val_.find(key)->second;
	return "";
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
      unsigned int bare_size() const {
	return bare_.size();
      }

      /**
       * Return the specified bare argument.
       *
       * @param n Bare argument position.
       * @return Bare argument at position.
       */
      std::string bare(unsigned int n) const {
	return bare_[n];
      }

      /**
       * Print a human readable parsed form of the command-line
       * arguments to the specified output stream.
       *
       * @param out Output stream.
       */
      void print(std::ostream& out) const {
	out << "COMMAND=" << cmd_ << '\n';
	unsigned int flag_count = 0;
	for (std::set<std::string>::const_iterator it = flag_.begin();
	     it != flag_.end();
	     ++it) {
	  out << "FLAG " << flag_count << "=" << (*it) << '\n';
	  ++flag_count;
	}
	unsigned int key_val_count = 0;
	for (std::map<std::string,std::string>::const_iterator it = key_val_.begin();
	     it != key_val_.end();
	     ++it) {
	  out << "KEY " << key_val_count << "=" << (*it).first;
	  out << " VAL " << key_val_count << "=" << (*it).second << '\n';
	  ++key_val_count;
	}
	unsigned int bare_count = 0;
	for (unsigned int i = 0; i < bare_.size(); ++i) {
	  out << "BARE ARG " << bare_count << "=" << bare_[i] << '\n';
	  ++bare_count;
	}
      }

    };


  }

}

#endif
