#ifndef __STAN__IO__CSV_WRITER_HPP__
#define __STAN__IO__CSV_WRITER_HPP__

#include <ostream>
#include <stan/math/matrix.hpp>

namespace stan {

  namespace io {

    /**
     * Writes Stan variables in comma-separated-value format to
     * an output stream.
     *
     * <p>All output is written to the same line and separated by commas.
     * Use the <code>newline()</code> method to advance to the next line
     * of CSV output.  
     */
    class csv_writer {
    private:
      std::ostream& o_;
      bool at_bol_;

    public: 

      /**
       * Write a comma.
       *
       * On the first call do nothing, subsequently write
       * a comma.
       */
      void comma() {
        if (at_bol_) {
          at_bol_ = false;
          return;
        }
        o_ << ",";
      }


      /**
       * Construct a CSV writer that writes to the specified
       * output stream.
       *
       * @param o Output stream on which to write.
       */
      csv_writer(std::ostream& o)
        : o_(o), at_bol_(true) {
      }

      /**
       * Write a newline character.
       *
       * End a line of output and start a new line.  This
       * method needs to be called rather than writing 
       * a newline to the output stream because it resets
       * the comma flag.
       */
      void newline() {
        o_ << "\n";
        at_bol_ = true;
      }

      /**
       * Write a value.
       * 
       * Write the specified integer to the output stream
       * on the current line.
       *
       * @param n Integer to write.
       */
      void write(int n) {
        comma();
        o_ << n;
      }

      /**
       * Write a value.
       * 
       * Write the specified double to the output stream
       * on the current line.
       *
       * @param x Double to write.
       */
      void write(double x) {
        comma();
        o_ << x;
      }

      /**
       * Write a value.
       * 
       * Write the specified vector to the output stream
       * on the current line.
       *
       * @param v Vector of doubles to write.
       */
      void write(const Eigen::Matrix<double,Eigen::Dynamic,1>& v) {
        typedef Eigen::Matrix<double,Eigen::Dynamic,1>::size_type size_type;
        for (size_type i = 0; i < v.size(); ++i)
          write(v[i]);
      }
      
      /**
       * Write a value.
       * 
       * Write the specified row vector to the output stream
       * on the current line.
       *
       * @param rv Row vector of doubles to write.
       */
      void write(const Eigen::Matrix<double,1,Eigen::Dynamic>& rv) {
        typedef Eigen::Matrix<double,1,Eigen::Dynamic>::size_type size_type;
        for (size_type i = 0; i < rv.size(); ++i)
          write(rv[i]);
      }

      /**
       * Write a value.
       * 
       * Write the specified matrix to the output stream in
       * row-major order.
       *
       * @param m Matrix to write.
       */
      void write(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& m) {
        typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
        for (size_type i = 0; i < m.rows(); ++i)
          for (size_type j = 0; j < m.cols(); ++j)
            write(m(i,j));
      }

      /**
       * Write a value in column-major order.
       * 
       * Write the specified matrix to the output stream in
       * column-major order.
       *
       * @param m Matrix of doubles to write.
       */
      void write_col_major(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& m) {
        typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
        for (size_type j = 0; j < m.cols(); ++j)
          for (size_type i = 0; i < m.rows(); ++i)
            write(m(i,j));
      }

      /**
       * Write a value.
       * 
       * Write the specified string to the output stream with
       * quotes around it and quotes inside escaped to double
       * quotes.
       *
       * @param s String to write.
       */
      void write(const std::string& s) {
        comma();

        o_ << '"';
        for (size_t i = 0; i < s.size(); ++i) {
          if (s.at(i) == '"') {
            o_ << '"' << '"'; // double quotes
          } else {
            o_ << s.at(i);
          }
        }
        o_ << '"';
      }


    };

  }

}

#endif
