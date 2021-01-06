#ifndef M_MATRIX_H_INCLUDED 
#define M_MATRIX_H_INCLUDED 

#include<eigen3/Eigen/Sparse> 
#include<vector> 
#include<eigen3/unsupported/Eigen/KroneckerProduct>
#include<eigen3/Eigen/Dense> 
#include<cmath>
#include<iostream>
typedef Eigen::SparseMatrix<double> SpMat; 
typedef Eigen::Triplet<double> T;
 
inline const int n2ind(const int n1, const int n2, const int N) {
    return n1 * N + n2; 
}

/** Reveal the content of a triplet list
  * \param t_list a vector of type T
**/
void reveal_triplet_list(const std::vector<T> &  t_list);

/** Generate the tau free part of m matrix, see python counterpart
 * 
 * 
 */
void get_m_matrix_tau_free(const int Nt, const int N, const double hat_t, SpMat& s1, SpMat& s2);

void get_m_matrix_delta_tau(const int Nt, const int N, const double hat_U, SpMat& delta_tau);

void get_m_matrix_same_4_all(const int Nt, const int N, const double hat_t, const double hat_U, SpMat& ret);

void get_m_matrix_xi(const double hat_u, Eigen::VectorXd& xi, SpMat& ret);

#endif