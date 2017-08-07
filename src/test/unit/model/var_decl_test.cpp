#include <stan/model/var_decl.hpp>
#include <test/unit/util.hpp>

TEST(var_decl, one) {
  using stan::model::var_decl;
  var_decl d("foo", "bar", 2);
  EXPECT_EQ("foo", d.name());
  EXPECT_EQ("bar", d.type_name());
  EXPECT_EQ(2, d.array_dims());
}
