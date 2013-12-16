#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <gtest/gtest.h>

#include <sstream>

#include <test/mcmc/hmc/integrators/models/command.cpp>
#include <test/mcmc/hmc/integrators/models/gauss.cpp>

#include <stan/io/dump.hpp>

#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG


// namespace
//************************************************************

typedef boost::ecuyer1988 rng_t;

class McmcHmcIntegratorsExplLeapfrogF : public testing::Test {
public:
  
  void SetUp() {
    static const std::string DATA = "mu <- 0.0\ny <- 0\n";
    std::stringstream data_stream(DATA);
    // setup hamiltonian
    stan::io::dump data_var_context(data_stream);

    model = new command_namespace::command(data_var_context);
  }
  
  void TearDown() {
    delete(model);
  }
  
  // integrator under test
  stan::mcmc::expl_leapfrog<
    stan::mcmc::unit_e_metric<command_namespace::command,rng_t>, 
    stan::mcmc::unit_e_point> unit_e_integrator;

  stan::mcmc::expl_leapfrog<
    stan::mcmc::diag_e_metric<command_namespace::command,rng_t>, 
    stan::mcmc::diag_e_point> diag_e_integrator;
  
  // model
  command_namespace::command *model;
};



TEST_F(McmcHmcIntegratorsExplLeapfrogF, begin_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);
  
  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.begin_update_p(z, hamiltonian, 0.5 * epsilon);
  EXPECT_NEAR(z.V,     1.99974742955684, 5e-14);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, update_q) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -1.68611660683688;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.update_q(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.99974742955684, 5e-14);
  EXPECT_NEAR(z.q(0),  1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, end_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.39887860643153;
  z.q(0) =  1.67264975797776;
  z.p(0) = -1.68611660683688;
  z.g(0) =  1.67264975797776;
  EXPECT_NEAR(z.V,     1.39887860643153, 1e-15);
  EXPECT_NEAR(z.q(0),  1.67264975797776, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0),  1.67264975797776, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.end_update_p(z, hamiltonian, 0.5 * epsilon);
  EXPECT_NEAR(z.V,     1.39887860643153, 5e-14);
  EXPECT_NEAR(z.q(0),  1.67264975797776, 5e-14);
  EXPECT_NEAR(z.p(0), -1.76974909473577, 5e-14);
  EXPECT_NEAR(z.g(0),  1.67264975797776, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_1) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.67676034808195, 5e-14);
  EXPECT_NEAR(z.q(0),  1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.77767970934226, 5e-14);
  EXPECT_NEAR(z.g(0),  1.83126205010749, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_2) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) =  0.531143888645192;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q(0),  2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0),  0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0),  2.06610501430439, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_3) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) =  0.531143888645192;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q(0),  2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0),  0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0),  2.06610501430439, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_4) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -1.01150787313287;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.01150787313287, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.4;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.03001529319458, 5e-14);
  EXPECT_NEAR(z.q(0),  1.43528066467474, 5e-14);
  EXPECT_NEAR(z.p(0), -1.69853874822605, 5e-14);
  EXPECT_NEAR(z.g(0),  1.43528066467474, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_5) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -0.141638464197442;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -0.141638464197442, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.8;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     0.777009958583946, 5e-14);
  EXPECT_NEAR(z.q(0),  1.24660335198005, 5e-14);
  EXPECT_NEAR(z.p(0), -1.44022928930593, 5e-14);
  EXPECT_NEAR(z.g(0),  1.24660335198005, 5e-14);
}
TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_6) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) =  0.0942249427134016;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.0942249427134016, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 1.6;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     0.0837242558054816, 5e-14);
  EXPECT_NEAR(z.q(0), -0.409204730680088, 5e-14);
  EXPECT_NEAR(z.p(0), -1.17831024137547, 5e-14);
  EXPECT_NEAR(z.g(0), -0.409204730680088, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_7) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) =  1.01936184962275;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  1.01936184962275, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 3.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     12.3878614837537, 5e-13);
  EXPECT_NEAR(z.q(0),  -4.97752176966686, 5e-14);
  EXPECT_NEAR(z.p(0),  5.78359874382383, 5e-14);
  EXPECT_NEAR(z.g(0),  -4.97752176966686, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_8) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V    =  1.99974742955684;
  z.q(0) =  1.99987371079118;
  z.p(0) = -2.73131279771964;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0),  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -2.73131279771964, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = -1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     6.96111198693627, 5e-14);
  EXPECT_NEAR(z.q(0),  3.73124965311523, 5e-14);
  EXPECT_NEAR(z.p(0),  0.134248884233563, 5e-14);
  EXPECT_NEAR(z.g(0),  3.73124965311523, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_9) {
  // setup z
  stan::mcmc::diag_e_point z(1);
  z.V    =  0.807684865121721;
  z.q(0) =  1.27097196280777;
  z.p(0) = -0.159996782671291;
  z.g(0) =  1.27097196280777;
  z.mInv(0) = 0.733184698671436;
  EXPECT_NEAR(z.V,     0.807684865121721, 1e-15);
  EXPECT_NEAR(z.q(0),  1.27097196280777, 1e-15);
  EXPECT_NEAR(z.p(0), -0.159996782671291, 1e-15);
  EXPECT_NEAR(z.g(0),  1.27097196280777, 1e-15);

  // setup hamiltonian
  stan::mcmc::diag_e_metric<command_namespace::command,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 2.40769920051673;

  diag_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.46626604258356, 5e-14);
  EXPECT_NEAR(z.q(0), -1.71246374711032, 5e-14);
  EXPECT_NEAR(z.p(0),  0.371492925378682, 5e-14);
  EXPECT_NEAR(z.g(0), -1.71246374711032, 5e-14);
}

TEST(McmcHmcIntegratorsExplLeapfrog, energy_conservation) {
  
  rng_t base_rng(0);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  gauss_namespace::gauss model(data_var_context, &std::cout);
  
  stan::mcmc::expl_leapfrog<
  stan::mcmc::unit_e_metric<gauss_namespace::gauss, rng_t>,
  stan::mcmc::unit_e_point> integrator;
  
  stan::mcmc::unit_e_metric<gauss_namespace::gauss, rng_t> metric(model, &std::cout);
  
  stan::mcmc::unit_e_point z(1);
  z.q(0) = 1;
  z.p(0) = 1;
  
  metric.update(z);
  double H0 = metric.H(z);
  double aveDeltaH = 0;
  
  double epsilon = 1e-3;
  double tau = 6.28318530717959;
  size_t L = tau / epsilon;
  
  for (size_t n = 0; n < L; ++n) {
    
    integrator.evolve(z, metric, epsilon);
    
    double deltaH = metric.H(z) - H0;
    aveDeltaH += (deltaH - aveDeltaH) / double(n + 1);
    
  }

  // Average error in Hamiltonian should be O(epsilon^{2})
  // in general, smaller for the gaussian case due to cancellations
  EXPECT_NEAR(aveDeltaH, 0, epsilon * epsilon);
  
}

TEST(McmcHmcIntegratorsExplLeapfrog, symplecticness) {
  
  rng_t base_rng(0);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  gauss_namespace::gauss model(data_var_context, &std::cout);
  
  stan::mcmc::expl_leapfrog<
  stan::mcmc::unit_e_metric<gauss_namespace::gauss, rng_t>,
  stan::mcmc::unit_e_point> integrator;
  
  stan::mcmc::unit_e_metric<gauss_namespace::gauss, rng_t> metric(model, &std::cout);
  
  // Create a circle of points
  const int n_points = 1000;
  
  double pi = 3.141592653589793;
  double r = 1.5;
  double q0 = 1;
  double p0 = 0;
  
  std::vector<stan::mcmc::unit_e_point> z;
  
  for (int i = 0; i < n_points; ++i) {
    z.push_back(stan::mcmc::unit_e_point(1));
    
    double theta = 2 * pi * (double)i / (double)n_points;
    z.back().q(0) = r * cos(theta) + q0;
    z.back().p(0)    = r * sin(theta) + p0;
  }
  
  // Evolve circle
  double epsilon = 1e-3;
  size_t L = pi / epsilon;
  
  for (int i = 0; i < n_points; ++i)
    metric.init(z.at(i));
  
  for (size_t n = 0; n < L; ++n)
    for (int i = 0; i < n_points; ++i)
      integrator.evolve(z.at(i), metric, epsilon);
  
  // Compute area of evolved shape using divergence theorem in 2D
  double area = 0;
  
  for (int i = 0; i < n_points; ++i) {
    
    double x1 = z[i].q(0);
    double y1 = z[i].p(0);
    double x2 = z[(i + 1) % n_points].q(0);
    double y2 = z[(i + 1) % n_points].p(0);
    
    double x_bary = 0.5 * (x1 + x2);
    double y_bary = 0.5 * (y1 + y2);
    
    double x_delta = x2 - x1;
    double y_delta = y2 - y1;
    
    double a = sqrt( x_delta * x_delta + y_delta * y_delta);
    
    double x_norm = 1;
    double y_norm = - x_delta / y_delta;
    double norm = sqrt( x_norm * x_norm + y_norm * y_norm );
    
    a *= (x_bary * x_norm + y_bary * y_norm) / norm;
    a = a < 0 ? -a : a;
    
    area += a;
    
  }
  
  area *= 0.5;
  
  // Symplectic integrators preserve volume (area in 2D)
  EXPECT_NEAR(area, pi * r * r, 1e-2);
  
}

