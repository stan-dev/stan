#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <test/test-models/no-main/mcmc/hmc/hamiltonians/funnel.cpp>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(BaseHamiltonian, update) {

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  funnel_model_namespace::funnel_model model(data_var_context, &std::cout);
  
  stan::mcmc::mock_hamiltonian<funnel_model_namespace::funnel_model, rng_t> metric(model, &std::cout);
  stan::mcmc::ps_point z(11);
  z.q.setOnes();
  
  metric.update(z);

  EXPECT_FLOAT_EQ(10.73223197, z.V);

  EXPECT_FLOAT_EQ(8.757758279, z.g(0));
  for (int i = 1; i < z.q.size(); ++i)
    EXPECT_FLOAT_EQ(0.1353352832, z.g(i));
  
}
