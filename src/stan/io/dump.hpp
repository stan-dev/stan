#ifndef __STAN__IO__DUMP_HPP__
#define __STAN__IO__DUMP_HPP__

#include <stdexcept>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <iostream>
#include <locale>

#include <boost/throw_exception.hpp>

#include <Eigen/Dense>

#include <stan/io/var_context.hpp>

namespace stan {

  namespace io {

    namespace {
       unsigned int product(std::vector<unsigned int> dims) {
         unsigned int y = 1U;
         for (unsigned int i = 0; i < dims.size(); ++i)
           y *= dims[i];
         return y;
       }
    }


    class dump_writer {
    private:
      std::ostream& out_;

      // checks doesn't contain quote char
      void write_name_equals(const std::string& name) {
        for (unsigned int i = 0; i < name.size(); ++i)
          if (name.at(i) == '"')
            BOOST_THROW_EXCEPTION(
              std::invalid_argument ("name can not contain quote char"));
        out_ << '"' << name << '"' << " <- " << '\n';
      }


      // adds period at end of output if necessary for double
      void write_val(const double& x) {
        std::stringstream ss;
        ss << x;
        std::string s;
        ss >> s;
        for (std::string::iterator it = s.begin();
             it < s.end();
             ++it) {
          if (*it == '.' || *it == 'e' || *it == 'E') {
            out_ << s;
            return;
          }
        }
        out_ << s << ".";
      }

      void write_val(const int& n) {
        out_ << n;
      }


      template <typename T>
      void write_list(T xs) {
        out_ << "c(";
        for (unsigned int i = 0; i < xs.size(); ++i) {
          if (i > 0) out_ << ", ";
          write_val(xs[i]);
        }
        out_ << ")";
      }

      template <typename T>
      void write_structure(std::vector<T> xs,
                           std::vector<unsigned int> dims) {
        out_ << "structure(";
        write_list(xs);
        out_ << ',' << '\n';
        out_ << ".Dim = ";
        write_list(dims);
        out_ << ")";
      }
      

      void dims(double x, std::vector<unsigned int> ds) {
        // no op
      }

      void dims(int x, std::vector<unsigned int> ds) {
        // no op
      }

      template <typename T> 
      void dims(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> m, 
                std::vector<unsigned int> ds) {
        ds.push_back(m.rows());
        ds.push_back(m.cols());
      }

      template <typename T> 
      void dims(Eigen::Matrix<T,Eigen::Dynamic,1> v, 
                std::vector<unsigned int> ds) {
        ds.push_back(v.size());
      }

      template <typename T> 
      void dims(Eigen::Matrix<T,1,Eigen::Dynamic> rv, 
                std::vector<unsigned int> ds) {
        ds.push_back(rv.size());
      }

      template <typename T> 
      void dims(std::vector<T> x, std::vector<unsigned int> ds) {
        ds.push_back(x.size());
        if (x.size() > 0)
          dims(x[0],ds);
      }
      
      template <typename T>
      std::vector<unsigned int> dims(T x) {
        std::vector<unsigned int> ds;
        dims(x,ds);
        return ds;
      }

      bool increment(const std::vector<unsigned int>& dims,
                     std::vector<unsigned int>& idx) {
        for (unsigned int i = 0; i < dims.size(); ++i) {
          ++idx[i];
          if (idx[i] < dims[i]) return true;
          idx[i] = 0;
        }
        return false;
      }

      template <typename T>
      void write_stan_val(const std::vector<T>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        unsigned int next_pos = pos + 1;
        write_stan_val(x[idx[pos]],idx,next_pos);
      }
      void write_stan_val(const std::vector<double>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        write_val(x[idx[pos]]);
      }
      void write_stan_val(const std::vector<int>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        write_val(x[idx[pos]]);
      }
      void write_stan_val(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        unsigned int next_pos = pos + 1;
        write_val(x(idx[pos],idx[next_pos]));
      }
      void write_stan_val(const Eigen::Matrix<double,1,Eigen::Dynamic>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        write_val(x[idx[pos]]);
      }
      void write_stan_val(const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                          const std::vector<unsigned int>& idx,
                          const unsigned int pos) {
        write_val(x[idx[pos]]);
      }


      template <typename T>
      void write_stan(const std::vector<T>& x) {
        std::vector<unsigned int> dims = dims(x);
        out_ << "structure(c(";
        std::vector<unsigned int> idx(dims.size(),0U);
        for (unsigned int count = 0; true; ++count) {
          if (count > 0) out_ << ", ";
          write_stan_val(x,idx);
          if (!increment(dims,idx)) break;
        }
        out_ << "), .Dim = ";
        write_list(dims);
        out_ << ")";
      }
      void write_stan(const std::vector<double>& x) {
        write_list(x);
      }
      void write_stan(const std::vector<int>& x) {
        write_list(x);
      }
      void write_stan(double x) {
        write_val(x);
      }
      void write_stan(int x) {
        write_val(x);
      }
      void write_stan(const Eigen::Matrix<double,1,Eigen::Dynamic>& x) {
        write_list(x);
      }
      void write_stan(const Eigen::Matrix<double,Eigen::Dynamic,1>& x) {
        write_list(x);
      }
      void write_stan(const Eigen::Matrix<double,
                                          Eigen::Dynamic,Eigen::Dynamic>& x) {
        out_ << "structure(c(";
        std::vector<double> vals;
        for (unsigned int m = 0; m < x.cols(); ++m) {
          for (unsigned int n = 0; n < x.rows(); ++n) {
            if (m > 0 || n > 0) out_ << ", ";
            write_val(x(m,n));
          }
        }
        out_ << "), .Dim = c(";
        out_ << x.rows() << ", " << x.cols();
        out_ << "))";
      }
      
    public:

      /**
       * Construct a dump writer writing to standard output.
       */
      dump_writer() : out_(std::cout) { }
      
      /**
       * Construct a dump writer writing to the specified output
       * stream.
       *
       * @param out Output stream for writing.
       */
      dump_writer(std::ostream& out) : out_(out) { }

      /**
       * Destroy this writer.
       */
      ~dump_writer() { }

      /**
       * Write a variable with the specified name and value.
       *
       * <p>This method will work for basic types consisting of
       * doubles and integers, Eigen types for vectors, row
       * vectors and matrices.  If this method works for a type,
       * it will work for a standard vector of that type.
       *
       * <p>Information concerning final type organization
       * into Eigen vectors, row vectors and matrices is lost.
       *
       * <p>All data is written in final-index dominant
       * ordering.
       * 
       * @param name Name of variable to write.
       * @param x Value of variable.
       * @tparam Type of variable being written.
       */
      template <typename T>
      void write(const std::string& name,
                 const T& x) {
        write_name_equals(name);
        write_stan(x);
      }

      /**
       * Write a structure variable with the specified name,
       * dimensions, and integer or double values.
       *
       * @param name Name of variable.
       * @param dims Dimensions of variable.
       * @param xs Values of variable in last-index major format.
       * @tparam T <code>double</code> or <code>int</code>.
       */
      template <typename T>
      void dump_structure(std::string name,
                          std::vector<unsigned int> dims,
                          std::vector<T> xs) {
        if (xs.size() != product(dims)) 
          BOOST_THROW_EXCEPTION(
              std::invalid_argument("xs.size() != product(dims)"));
        write_structure(xs,dims);
      }

      /**
       * Write a list variable with the specified name and
       * integer or double values.
       * 
       * @param name Name of variable.
       * @param xs Values of variable.
       * @tparam T <code>double</code> or <code>int</code>.
       */

      template <typename T>
      void dump_list(std::string name,
                     std::vector<T> xs) {
        write_name_equals(name);
        write_list(xs);
      }

      /**
       * Write a list variable with the specified name and
       * integer or double values.
       * 
       * @param name Name of variable.
       * @param x Value of variable.
       * @tparam T <code>double</code> or <code>int</code>.
       */
      template <typename T>
      void dump_var(std::string name,
                    T x) {
        write_name_equals(name);
        write_val(x);
      }
      
    };

    /**
     * A <code>dump_reader</code> parses data from the S-plus dump
     * format, a human-readable ASCII representation of arbitrarily
     * dimensioned arrays of integers and arrays of floating point
     * values.
     *
     * <p>Stan's dump reader is limited to reading
     * integers, scalars and arrays of arbitrary dimensionality of
     * integers and scalars.  It is able to read from a file
     * consisting of a sequence of dumped variables.
     *
     * <p>There cannot be any <code>NA</code>
     * (i.e., undefined) values, because these cannot be
     * represented as <code>double</code> values.
     *
     * <p>The dump reader class follows a standard scanner pattern.
     * The method <code>next()</code> is called to scan the next
     * input.  The type, dimensions, and values of the input is then
     * available through method calls.  Here, the type is either
     * double or integer, and the values are the name of the variable
     * and its array of values.  If there is a single value, the
     * dimension array is empty.  For a list, the dimension
     * array contains a single entry for the number of values.  
     * For an array, the dimensions are the dimensions of the array.
     *
     * <p>Reads are performed in an "S-compatible" mode whereby
     * a string such as "1" or "-127" denotes and integer, whereas
     * a string such as "1." or "0.9e-5" represents a floating
     * point value.  
     *
     * <p>For dumping, arrays are indexed in last-index major fashion,
     * which corresponds to column-major order for matrices
     * represented as two-dimensional arrays.  As a result, the first
     * indices change fastest.  For instance, if there is an
     * three-dimensional array <code>x</code> with dimensions
     * <code>[2,2,2]</code>, then there are 8 values, provided in the
     * order
     *
     * <p><code>[0,0,0]</code>, 
     * <code>[1,0,0]</code>, 
     * <code>[0,1,0]</code>, 
     * <code>[1,1,0]</code>, 
     * <code>[0,0,1]</code>, 
     * <code>[1,0,1]</code>, 
     * <code>[0,1,1]</code>, 
     * <code>[1,1,1]</code>.
     *
     */
    class dump_reader {
    private:
      std::string name_;
      std::vector<int> stack_i_;
      std::vector<double> stack_r_;
      std::vector<unsigned int> dims_;
      std::istream& in_;


      bool scan_char(char c_expected) {
        char c;
        in_ >> c;
        if (c != c_expected) {
          in_.putback(c);
          return false;
        }
        return true;
      }

      bool scan_name_unquoted() {
        char c;
        in_ >> c; // 
        if (!std::isalpha(c)) return false;
        name_.push_back(c); 
        while (in_.get(c)) { // get turns off auto space skip
          if (std::isalpha(c) || std::isdigit(c) || c == '_' || c == '.') {
            name_.push_back(c);
          } else {
            in_.putback(c);
            return true;
          }
        }
        return true; // but hit eos
      }

      bool scan_name() {
        if (scan_char('"')) {
          if (!scan_name_unquoted()) return false;
          if (!scan_char('"')) return false;
        } else {
          if (!scan_name_unquoted()) return false;
        }
        return true;
      }

      bool scan_chars(std::string s) {
        for (unsigned int i = 0; i < s.size(); ++i) {
          char c;
          if (!(in_ >> c)) {
            for (unsigned int j = 1; j < i; ++j) 
              in_.putback(s[i-j]);
            return false;
          }
          if (c != s[i]) {
            in_.putback(c);
            for (unsigned int j = 1; j < i; ++j) 
              in_.putback(s[i-j]);
            return false;
          }
        }
        return true;
      }

      bool scan_number() {
        std::string buf;
        bool is_double = false;
        char c;
        while (in_.get(c)) {
          if (std::isspace(c)) continue;
          in_.putback(c);
          break;
        }
        while (in_.get(c)) {
          if (std::isspace(c)) continue;
          if (std::isdigit(c) || c == '-') {
            buf.push_back(c);
          } else if (c == '.'
                     || c == 'e'
                     || c == 'E') {
            is_double = true;
            buf.push_back(c);
          }
          else {
            in_.putback(c);
            break;
          }
        }
        if (!is_double && stack_r_.size() == 0) {
          int n;
          if (!(std::stringstream(buf) >> n))
            return false;
          stack_i_.push_back(n);
        } else {
          for (unsigned int j = 0; j < stack_i_.size(); ++j)
            stack_r_.push_back(static_cast<double>(stack_i_[j]));
          stack_i_.clear();
          double x;
          if (!(std::stringstream(buf) >> x))
            return false;
          stack_r_.push_back(x);
        }
        return true;
      }

      void print_next_char() {
        char c;
        bool ok = in_.get(c);
        if (ok) {
          std::cout << "next char=" << c << std::endl;
          in_.putback(c);
        } else {
          std::cout << "next char=<EOS>" << std::endl;
        }
      }

      bool scan_seq_value() {
        if (!scan_char('(')) return false;
        if (scan_char(')')) {
          dims_.push_back(0U);
          return true;
        }
        if (!scan_number()) return false;; // first entry
        while (scan_char(',')) {
          if (!scan_number()) return false;
        }
        dims_.push_back(stack_r_.size() + stack_i_.size());
        return scan_char(')');
      }

      bool scan_struct_value() {
        if (!scan_char('(')) return false;
        if (!scan_char('c')) return false;
        scan_seq_value();
        dims_.clear();
        if (!scan_char(',')) return false;
        if (!scan_char('.')) return false;
        if (!scan_chars("Dim")) return false;
        if (!scan_char('=')) return false;
        if (!scan_char('c')) return false;
        if (!scan_char('(')) return false;
        unsigned int dim;
        in_ >> dim;
        dims_.push_back(dim);
        while (scan_char(',')) {
          in_ >> dim;
          dims_.push_back(dim);
        }
        if (!scan_char(')')) return false;
        if (!scan_char(')')) return false;
        return true;
      }

  

      bool scan_value() {
        if (scan_char('c'))
          return scan_seq_value();
        if (scan_chars("structure"))
          return scan_struct_value();
        return scan_number();
      }

    public:
      /**
       * Construct a reader for standard input.
       */
      dump_reader() : in_(std::cin) { }

      /**
       * Construct a reader for the specified input stream.
       *
       * @param in Input stream reference from which to read.
       */
      dump_reader(std::istream& in) : in_(in) { }

      /**
       * Destroy this reader.
       */
      ~dump_reader() { }


      /**
       * Return the name of the most recently read variable.
       *
       * @return Name of most recently read variable.
       */
      std::string name() {
        return name_;
      }

      /**
       * Return the dimensions of the most recently
       * read variable.
       *
       * @return Last dimensions.
       */
      std::vector<unsigned int> dims() {
        return dims_;
      }

      /**
       * Return <code>true</code> if the value(s) in the most recently
       * read item are integer values and <code>false</code> if
       * they are floating point.
       */
      bool is_int() {
        return stack_i_.size() > 0;
      }

      /**
       * Returns the integer values from the last item if the
       * last item read was an integer and the empty vector otherwise.
       *
       * @return Integer values of last item.
       */
      std::vector<int> int_values() {
        return stack_i_;
      }

      /**
       * Returns the floating point values from the last item if the
       * last item read contained floating point values and the empty
       * vector otherwise.
       *
       * @return Floating point values of last item.
       */
      std::vector<double> double_values() {
        return stack_r_;
      }

      /**
       * Read the next value from the input stream, returning
       * <code>true</code> if successful and <code>false</code> if no
       * further input may be read.
       *
       * @return Return <code>true</code> if a fresh variable was read.
       */
      bool next() {
        stack_r_.clear();
        stack_i_.clear();
        dims_.clear();
        name_.erase();
        if (!scan_name()) return false;
        if (!scan_char('<')) return false;
        if (!scan_char('-')) return false;
        if (!scan_value()) return false;
        return true;
      }

      void print() {
        std::cout << "var name=|" << name_ << "|" << std::endl;
        std:: cout << "dims=(";
        for (unsigned int i = 0; i < dims_.size(); ++i) {
          if (i > 0)
            std::cout << ",";
          std::cout << dims_[i];
        }
        std::cout << ")" << std::endl;
        std::cout << "float stack:" << std::endl;
        for (unsigned int i = 0; i < stack_r_.size(); ++i)
          std::cout << "  [" << i << "] " << stack_r_[i] << std::endl;
        std::cout << "int stack" << std::endl;
        for (unsigned int i = 0; i < stack_i_.size(); ++i)
          std::cout << "  [" << i << "] " << stack_i_[i] << std::endl;
      }

  
    };



    /**
     * A dump object represents a dump of named arrays with dimensions.
     * The arrays may have any dimensionality.  The values for an array
     * are typed to double or int.  
     *
     * <p>See <code>dump_reader</code> for more information on the format.
     *
     * <p>Dump objects are created from reading dump files from an
     * input stream.  
     *
     * <p>The dimensions and values of variables
     * may be accessed by name. 
     */
    class dump : public stan::io::var_context {
    private: 
      std::map<std::string, 
               std::pair<std::vector<double>,
                         std::vector<unsigned int> > > vars_r_;
      std::map<std::string, 
               std::pair<std::vector<int>, 
                         std::vector<unsigned int> > > vars_i_;
      std::vector<double> const empty_vec_r_;
      std::vector<int> const empty_vec_i_;
      std::vector<unsigned int> const empty_vec_ui_;
      /**
       * Return <code>true</code> if this dump contains the specified
       * variable name is defined in the real values. This method
       * returns <code>false</code> if the values are all integers.
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable exists in the 
       * real values of the dump.
       */
      bool contains_r_only(const std::string& name) const {
        return vars_r_.find(name) != vars_r_.end();
      }
    public: 

      /**
       * Construct a dump object from the specified input stream.
       *
       * <b>Warning:</b> This method does not close the input stream.
       *
       * @param in Input stream from which to read.
       */
      dump(std::istream& in) {
        dump_reader reader(in);
        while (reader.next()) {
          if (reader.is_int()) {
            vars_i_[reader.name()] 
              = std::pair<std::vector<int>, 
                          std::vector<unsigned int> >(reader.int_values(), 
                                                      reader.dims());
            
          } else {
            vars_r_[reader.name()] 
              = std::pair<std::vector<double>, 
                          std::vector<unsigned int> >(reader.double_values(), 
                                                      reader.dims());
          }
        }
      }

      /**
       * Return <code>true</code> if this dump contains the specified
       * variable name is defined. This method returns <code>true</code>
       * even if the values are all integers.
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable exists.
       */
      bool contains_r(const std::string& name) const {
        return contains_r_only(name) || contains_i(name);
      }

      /**
       * Return <code>true</code> if this dump contains an integer
       * valued array with the specified name.
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable name has an integer
       * array value.
       */
      bool contains_i(const std::string& name) const {
        return vars_i_.find(name) != vars_i_.end();
      }

      /**
       * Return the double values for the variable with the specified
       * name or null. 
       *
       * @param name Name of variable.
       * @return Values of variable.
       */
      std::vector<double> vals_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).first;
        } else if (contains_i(name)) {
          std::vector<int> vec_int = (vars_i_.find(name)->second).first;
          std::vector<double> vec_r(vec_int.size());
          for (unsigned int ii = 0; ii < vec_int.size(); ii++) {
            vec_r[ii] = vec_int[ii];
          }
          return vec_r;
        }
        return empty_vec_r_;
      }
      
      /**
       * Return the dimensions for the double variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Dimensions of variable.
       */
      std::vector<unsigned int> dims_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).second;
        } else if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }

      /**
       * Return the integer values for the variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Values.
       */
      std::vector<int> vals_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).first;
        }
        return empty_vec_i_;
      }
      
      /**
       * Return the dimensions for the integer variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Dimensions of variable.
       */
      std::vector<unsigned int> dims_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }

      /**
       * Return a list of the names of the floating point variables in
       * the dump.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_r(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<double>,
                                std::vector<unsigned int> > >
                 ::const_iterator it = vars_r_.begin();
             it != vars_r_.end(); ++it)
          names.push_back((*it).first);
      }

      /**
       * Return a list of the names of the integer variables in
       * the dump.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_i(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<int>, 
                                std::vector<unsigned int> > >
                 ::const_iterator it = vars_i_.begin();
             it != vars_i_.end(); ++it)
          names.push_back((*it).first);
      }

      bool remove(const std::string& name) {
        return (vars_i_.erase(name) > 0) 
          || (vars_r_.erase(name) > 0);
      }
      
    };
    

  }


}


#endif
