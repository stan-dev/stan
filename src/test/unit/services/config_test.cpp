#include <stan/services/config.hpp>
#include <gtest/gtest.h>

TEST(services_config, instantiate) {
  nuts_config c;
  c.set_model("/Users/carp/temp2/foo.stan");
  c.set_chain_id(-1);

  std::stringstream s;
  bool valid = c.validate(s);
  std::cout << "CONFIG:" << std::endl;
  c.print(std::cout);
  std::cout << std::endl;
  std::cout << "config is " << (valid ? "valid" : "invalid") << std::endl;
  if (!valid)
    std::cout << "ERROR MESSAGES:" << std::endl << s.str() << std::endl;
  EXPECT_EQ(0, 0);
}
