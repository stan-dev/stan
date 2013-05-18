// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARRAYWRAPPER_H
#define EIGEN_ARRAYWRAPPER_H

namespace Eigen { 

/** \class ArrayWrapper
  * \ingroup Core_Module
  *
  * \brief Expression of a mathematical vector or matrix as an array object
  *
  * This class is the return type of MatrixBase::array(), and most of the time
  * this is the only way it is use.
  *
  * \sa MatrixBase::array(), class MatrixWrapper
  */

namespace internal {
template<typename ExpressionType>
struct traits<ArrayWrapper<ExpressionType> >
  : public traits<typename remove_all<typename ExpressionType::Nested>::type >
{
  typedef ArrayXpr XprKind;
};
}

template<typename ExpressionType>
class ArrayWrapper : public ArrayBase<ArrayWrapper<ExpressionType> >
{
  public:
    typedef ArrayBase<ArrayWrapper> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(ArrayWrapper)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ArrayWrapper)

    typedef typename internal::conditional<
                       internal::is_lvalue<ExpressionType>::value,
                       Scalar,
                       const Scalar
                     >::type ScalarWithConstIfNotLvalue;

    typedef typename internal::nested<ExpressionType>::type NestedExpressionType;

    inline ArrayWrapper(ExpressionType& matrix) : m_expression(matrix) {}

    inline Index rows() const { return m_expression.rows(); }
    inline Index cols() const { return m_expression.cols(); }
    inline Index outerStride() const { return m_expression.outerStride(); }
    inline Index innerStride() const { return m_expression.innerStride(); }

    inline ScalarWithConstIfNotLvalue* data() { return m_expression.data(); }
    inline const Scalar* data() const { return m_expression.data(); }

    inline CoeffReturnType coeff(Index row, Index col) const
    {
      return m_expression.coeff(row, col);
    }

    inline Scalar& coeffRef(Index row, Index col)
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    inline const Scalar& coeffRef(Index row, Index col) const
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    inline CoeffReturnType coeff(Index index) const
    {
      return m_expression.coeff(index);
    }

    inline Scalar& coeffRef(Index index)
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    inline const Scalar& coeffRef(Index index) const
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index row, Index col) const
    {
      return m_expression.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(Index row, Index col, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index index) const
    {
      return m_expression.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(Index index, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

    template<typename Dest>
    inline void evalTo(Dest& dst) const { dst = m_expression; }

    const typename internal::remove_all<NestedExpressionType>::type& 
    nestedExpression() const 
    {
      return m_expression;
    }

    /** Forwards the resizing request to the nested expression
      * \sa DenseBase::resize(Index)  */
    void resize(Index newSize) { m_expression.const_cast_derived().resize(newSize); }
    /** Forwards the resizing request to the nested expression
      * \sa DenseBase::resize(Index,Index)*/
    void resize(Index nbRows, Index nbCols) { m_expression.const_cast_derived().resize(nbRows,nbCols); }

  protected:
    NestedExpressionType m_expression;
};

/** \class MatrixWrapper
  * \ingroup Core_Module
  *
  * \brief Expression of an array as a mathematical vector or matrix
  *
  * This class is the return type of ArrayBase::matrix(), and most of the time
  * this is the only way it is use.
  *
  * \sa MatrixBase::matrix(), class ArrayWrapper
  */

namespace internal {
template<typename ExpressionType>
struct traits<MatrixWrapper<ExpressionType> >
 : public traits<typename remove_all<typename ExpressionType::Nested>::type >
{
  typedef MatrixXpr XprKind;
};
}

template<typename ExpressionType>
class MatrixWrapper : public MatrixBase<MatrixWrapper<ExpressionType> >
{
  public:
    typedef MatrixBase<MatrixWrapper<ExpressionType> > Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(MatrixWrapper)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixWrapper)

    typedef typename internal::conditional<
                       internal::is_lvalue<ExpressionType>::value,
                       Scalar,
                       const Scalar
                     >::type ScalarWithConstIfNotLvalue;

    typedef typename internal::nested<ExpressionType>::type NestedExpressionType;

    inline MatrixWrapper(ExpressionType& matrix) : m_expression(matrix) {}

    inline Index rows() const { return m_expression.rows(); }
    inline Index cols() const { return m_expression.cols(); }
    inline Index outerStride() const { return m_expression.outerStride(); }
    inline Index innerStride() const { return m_expression.innerStride(); }

    inline ScalarWithConstIfNotLvalue* data() { return m_expression.data(); }
    inline const Scalar* data() const { return m_expression.data(); }

    inline CoeffReturnType coeff(Index row, Index col) const
    {
      return m_expression.coeff(row, col);
    }

    inline Scalar& coeffRef(Index row, Index col)
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    inline const Scalar& coeffRef(Index row, Index col) const
    {
      return m_expression.derived().coeffRef(row, col);
    }

    inline CoeffReturnType coeff(Index index) const
    {
      return m_expression.coeff(index);
    }

    inline Scalar& coeffRef(Index index)
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    inline const Scalar& coeffRef(Index index) const
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index row, Index col) const
    {
      return m_expression.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(Index row, Index col, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index index) const
    {
      return m_expression.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(Index index, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

    const typename internal::remove_all<NestedExpressionType>::type& 
    nestedExpression() const 
    {
      return m_expression;
    }

    /** Forwards the resizing request to the nested expression
      * \sa DenseBase::resize(Index)  */
    void resize(Index newSize) { m_expression.const_cast_derived().resize(newSize); }
    /** Forwards the resizing request to the nested expression
      * \sa DenseBase::resize(Index,Index)*/
    void resize(Index nbRows, Index nbCols) { m_expression.const_cast_derived().resize(nbRows,nbCols); }

  protected:
    NestedExpressionType m_expression;
};

} // end namespace Eigen

#endif // EIGEN_ARRAYWRAPPER_H
