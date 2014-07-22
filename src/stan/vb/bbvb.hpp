#ifndef __STAN__VB__BBVB__HPP__
#define __STAN__VB__BBVB__HPP__

#include <ostream>

#include <stan/vb/base_vb.hpp>
#include <stan/vb/latent_vars.hpp>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace vb {

    template <class M, class BaseRNG>
    class bbvb : public base_vb
    {

    public:

      bbvb(M& m, BaseRNG& rng, std::ostream* o, std::ostream* e):
        base_vb(o, e),
        model_(m), rng_(rng) {};

      virtual ~bbvb() {};

      void test()
      {
        if (out_stream_) *out_stream_ << "This is base_vb::bbvb::test()" << std::endl;

        Eigen::VectorXd mu = Eigen::VectorXd::Constant(4,1.35);
         Eigen::MatrixXd L = Eigen::MatrixXd::Identity(4,4);
        // L *= 2;
        // Eigen::MatrixXd L;
        L << 10, 0, 0, 0,
             9, 10, 0, 0,
             8, 7, 10, 0,
             1, 16, 6, 10;

        latent_vars asdf = latent_vars(mu,L);

        if (out_stream_) *out_stream_ << "asdf.mu() = " << std::endl
                                      << asdf.mu() << std::endl;

        if (out_stream_) *out_stream_ << "asdf.L() = " << std::endl
                                      << asdf.L() << std::endl;

        Eigen::VectorXd x = Eigen::VectorXd::Constant(4,10.0);

        if (out_stream_) *out_stream_ << "x = " << std::endl
                                      << x << std::endl;

        Eigen::VectorXd unconst = asdf.to_unconstrained(x);

        if (out_stream_) *out_stream_ << "asdf.to_unconstrained(x) = " << std::endl
                                      << unconst << std::endl;

        Eigen::VectorXd stand = asdf.to_standardized(unconst);

        if (out_stream_) *out_stream_ << "asdf.to_standardized(unconst) = " << std::endl
                                      << stand << std::endl;

      }

    protected:

      M& model_;
      BaseRNG& rng_;

      void write_error_msg_(std::ostream* error_msgs, const std::exception& e)
      {
        if (!error_msgs) return;

        *error_msgs << std::endl
                    << "[stan::vb::bbvb.hpp] encountered an error:"
                    << std::endl
                    << e.what() << std::endl << std::endl;
      }

    };

  } // vb

} // stan

#endif

