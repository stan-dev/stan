// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/AlignedVector3>

template<typename Scalar>
void alignedvector3()
{
  Scalar s1 = internal::random<Scalar>();
  Scalar s2 = internal::random<Scalar>();
  typedef Matrix<Scalar,3,1> RefType;
  typedef Matrix<Scalar,3,3> Mat33;
  typedef AlignedVector3<Scalar> FastType;
  RefType  r1(RefType::Random()), r2(RefType::Random()), r3(RefType::Random()),
           r4(RefType::Random()), r5(RefType::Random()), r6(RefType::Random());
  FastType f1(r1), f2(r2), f3(r3), f4(r4), f5(r5), f6(r6);
  Mat33 m1(Mat33::Random());
  
  VERIFY_IS_APPROX(f1,r1);
  VERIFY_IS_APPROX(f4,r4);

  VERIFY_IS_APPROX(f4+f1,r4+r1);
  VERIFY_IS_APPROX(f4-f1,r4-r1);
  VERIFY_IS_APPROX(f4+f1-f2,r4+r1-r2);
  VERIFY_IS_APPROX(f4+=f3,r4+=r3);
  VERIFY_IS_APPROX(f4-=f5,r4-=r5);
  VERIFY_IS_APPROX(f4-=f5+f1,r4-=r5+r1);
  VERIFY_IS_APPROX(f5+f1-s1*f2,r5+r1-s1*r2);
  VERIFY_IS_APPROX(f5+f1/s2-s1*f2,r5+r1/s2-s1*r2);
  
  VERIFY_IS_APPROX(m1*f4,m1*r4);
  VERIFY_IS_APPROX(f4.transpose()*m1,r4.transpose()*m1);
  
  VERIFY_IS_APPROX(f2.dot(f3),r2.dot(r3));
  VERIFY_IS_APPROX(f2.cross(f3),r2.cross(r3));
  VERIFY_IS_APPROX(f2.norm(),r2.norm());

  VERIFY_IS_APPROX(f2.normalized(),r2.normalized());

  VERIFY_IS_APPROX((f2+f1).normalized(),(r2+r1).normalized());
  
  f2.normalize();
  r2.normalize();
  VERIFY_IS_APPROX(f2,r2);
}

void test_alignedvector3()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( alignedvector3<float>() );
  }
}
