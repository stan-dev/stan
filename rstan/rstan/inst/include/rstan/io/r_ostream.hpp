
#ifndef __R__IO_R_OSTREAM_HPP__
#define __R__IO_R_OSTREAM_HPP__

#include <streambuf>
#include <ostream>
#include <Rinternals.h>

/*
 * Since Rcpp only provides a std::cout, similar version
 * of both std::cout and std::cerr are implemented 
 * for stan to write to cout and cerr of R.
 *
 * see http://goo.gl/mKmeP (or http://goo.gl/1AB66) 
 */ 

namespace rstan {
  namespace io {

    class r_cout_streambuf : public std::streambuf {
    public:
      r_cout_streambuf() {} 

    protected:
      virtual int_type overflow(int_type c) {
        if (c != EOF) {
          char z = c;
          Rprintf("%c", c);
          return EOF;
        }
        return c;
      }

      virtual std::streamsize xsputn(const char* s, std::streamsize n) {
        Rprintf("%.*s", n, s);
        return n;
      }
    };

    class r_cerr_streambuf : public std::streambuf {
    public:
      r_cerr_streambuf() {} 

    protected:
      virtual int_type overflow(int_type c) {
        if (c != EOF) {
          char z = c;
          REprintf("%c", c);
          return EOF;
        }
        return c;
      }

      virtual std::streamsize xsputn(const char* s, std::streamsize n) {
        REprintf("%.*s", n, s);
        return n;
      }
    };


    template <class T> class r_ostream : public std::ostream {
      protected:
        T buf;
      public:
        r_ostream() : std::ostream(&buf) {}
    }; 

    r_ostream<r_cout_streambuf> rcout; 
    r_ostream<r_cerr_streambuf> rcerr; 
  }

} 
#endif 
