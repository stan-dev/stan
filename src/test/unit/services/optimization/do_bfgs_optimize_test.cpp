#include <gtest/gtest.h>
#include <stan/services/optimization/do_bfgs_optimize.hpp>
#include <stan/optimization/bfgs.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/additive_combine.hpp>
#include <test/unit/util.hpp>

typedef rosenbrock_model_namespace::rosenbrock_model Model;
typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)

struct mock_callback {
  int n;
  mock_callback() : n(0) { }
  
  void operator()() {
    n++;
  }
};

TEST(Services, do_bfgs_optimize__bfgs) {
  typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer_BFGS;

  std::vector<double> cont_vector(2);
  cont_vector[0] = -1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  std::stringstream out;
  Optimizer_BFGS bfgs(model, cont_vector, disc_vector, &out);
  EXPECT_EQ("", out.str());

  double lp = 0;
  bool save_iterations = true;
  int refresh = 0;
  int return_code;
  unsigned int random_seed = 0;
  rng_t base_rng(random_seed);

  std::fstream* output_stream = 0;
  mock_callback callback;

  std::stringstream notice;
  return_code = stan::services::optimization::do_bfgs_optimize(model,bfgs, base_rng,
                                                               lp, cont_vector, disc_vector,
                                                               output_stream, &notice,
                                                               save_iterations, refresh,
                                                               callback);
  EXPECT_EQ("initial log joint probability = -4\nOptimization terminated normally: \n  Convergence detected: relative gradient magnitude is below tolerance\n", notice.str());
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(33, callback.n);
}
  
TEST(Services, do_bfgs_optimize__lbfgs) {
  std::vector<double> cont_vector(2);
  cont_vector[0] = -1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::LBFGSUpdate<> > Optimizer_LBFGS;
  std::stringstream out;
  Optimizer_LBFGS lbfgs(model, cont_vector, disc_vector, &out);
  EXPECT_EQ("", out.str());


  double lp = 0;
  bool save_iterations = true;
  int refresh = 0;
  int return_code;
  unsigned int random_seed = 0;
  rng_t base_rng(random_seed);

  std::fstream* output_stream = 0;
  mock_callback callback;

  std::stringstream notice;
  return_code = stan::services::optimization::do_bfgs_optimize(model, lbfgs, base_rng,
                                                               lp, cont_vector, disc_vector,
                                                               output_stream, &notice,
                                                               save_iterations, refresh,
                                                               callback);
  EXPECT_EQ("initial log joint probability = -4\nOptimization terminated normally: \n  Convergence detected: relative gradient magnitude is below tolerance\n", notice.str());
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(35, callback.n);
}

TEST(Services, do_bfgs_optimize__streams) {
  stan::test::capture_std_streams();
  
  std::vector<double> cont_vector(2);
  cont_vector[0] = -1; cont_vector[1] = 1;
  std::vector<int> disc_vector;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::LBFGSUpdate<> > Optimizer_LBFGS;
  Optimizer_LBFGS lbfgs_none(model, cont_vector, disc_vector, 0);

  std::stringstream out;
  Optimizer_LBFGS lbfgs_out(model, cont_vector, disc_vector, &out);
  EXPECT_EQ("", out.str());


  double lp = 0;
  bool save_iterations = true;
  int refresh = 0;
  unsigned int random_seed = 0;
  rng_t base_rng(random_seed);

  mock_callback callback;

  EXPECT_NO_THROW(stan::services::optimization::do_bfgs_optimize(model, lbfgs_none, base_rng,
                                                                 lp, cont_vector, disc_vector,
                                                                 0, 0,
                                                                 save_iterations, refresh,
                                                                 callback));

  std::stringstream notice;
  out.str("");
  EXPECT_NO_THROW(stan::services::optimization::do_bfgs_optimize(model, lbfgs_out, base_rng,
                                                                 lp, cont_vector, disc_vector,
                                                                 &out, &notice,
                                                                 save_iterations, refresh,
                                                                 callback));
  EXPECT_EQ("-4,1,1\n-3.99039,-0.996,1\n-3.98595,-0.993977,0.997991\n-3.95108,-0.977795,0.975935\n-3.86719,-0.939958,0.915732\n-2.53443,-0.566307,0.292223\n-2.52633,-0.548405,0.264863\n-2.03821,-0.37804,0.105603\n-1.66199,-0.287146,0.0752085\n-1.49873,-0.16389,-0.0110988\n-1.41698,-0.143505,-0.0124782\n-1.06195,-0.0130123,-0.0187393\n-0.844793,0.0925618,-0.00604355\n-0.594534,0.230199,0.0573958\n-0.522564,0.299313,0.0718113\n-0.482979,0.330783,0.0906752\n-0.319026,0.464684,0.197914\n-0.224935,0.526,0.278284\n-0.170516,0.623674,0.371971\n-0.156839,0.632555,0.385353\n-0.0909244,0.735235,0.526139\n-0.0642486,0.747222,0.560215\n-0.0474154,0.782278,0.611604\n-0.0302953,0.861901,0.732279\n-0.0268654,0.861405,0.733268\n-0.00902424,0.907354,0.821191\n-0.00406796,0.948396,0.895708\n-0.00055095,0.976576,0.953851\n-0.00033189,1.0002,0.998576\n-6.84319e-05,0.996626,0.992507\n-2.95128e-06,0.998365,0.99668\n-2.35946e-07,0.999518,0.999042\n-4.02691e-10,0.999987,0.999975\n-1.22832e-10,0.999989,0.999978\n-6.84734e-11,0.999992,0.999983\n-8.67059e-19,1,1\n",
            out.str());
  EXPECT_EQ("initial log joint probability = -4\nOptimization terminated normally: \n  Convergence detected: relative gradient magnitude is below tolerance\n", notice.str());

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
