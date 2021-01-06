#include "m_matrix.h"


void reveal_triplet_list(const std::vector<T> &  t_list) {
    for(auto& x: t_list){
        std::cout << x.row() << ' ' << x.col() << ' ' << ' ' << x.value() << '\n';
    }
    std::cout << '\n';
}


void get_m_matrix_tau_free(const int Nt, const int N, const double hat_t, SpMat& s1, SpMat& s2) {
    std::vector<T> tripletList;

    for (int n1=0; n1 < N; ++n1)
        for (int n2 = 0; n2 < N; ++n2)
        {
            auto y = n2ind(n1, n2, N); 
            tripletList.push_back(T(y, y,-hat_t));
            tripletList.push_back(T(n2ind(n1, (n2+1)%N, N), y, -hat_t/2.));
            tripletList.push_back(T(n2ind(n1, (n2-1+N)%N, N), y, -hat_t/2.));
            tripletList.push_back(T(n2ind((n1+1)%N, n2, N), y, -hat_t/2.));
            tripletList.push_back(T(n2ind((n1-1+N)%N, n2, N), y, -hat_t/2.));
        }
        
    SpMat buf1(N*N, N*N); 
    buf1.setFromTriplets(tripletList.begin(), tripletList.end());
    SpMat prop2s1 = Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(Nt, Nt), buf1);

    tripletList.clear();

    for (int n1=0; n1 < N; ++n1)
        for (int n2 = 0; n2 < N; ++n2)
        {
            auto y = n2ind(n1, n2, N); 
            tripletList.push_back(T(n2ind(n1, (n2+1)%N, N), y, hat_t/2.));
            tripletList.push_back(T(n2ind(n1, (n2-1+N)%N, N), y, -hat_t/2.));
            tripletList.push_back(T(n2ind((n1+1)%N, n2, N), y, hat_t/2.));
            tripletList.push_back(T(n2ind((n1-1+N)%N, n2, N), y, -hat_t/2.));
        }
    
    SpMat buf2(N*N, N*N); 
    buf2.setFromTriplets(tripletList.begin(), tripletList.end());
    SpMat prop2s2 = Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(Nt, Nt), buf2);
     
    s1 = prop2s1 + prop2s2;
    s2 = prop2s1 - prop2s2;
}

void get_m_matrix_delta_tau(const int Nt, const int N, const double hat_U, SpMat& delta_tau) {
    std::vector<T> tripletList;
    int mat_size = N * N * Nt; 
    for (int y = 0; y < N*N; ++y)
    {
        tripletList.push_back(T(y-N*N+mat_size, y,-1.));
        tripletList.push_back(T(y,y,-1.));
        tripletList.push_back(T(y,y,hat_U));
        tripletList.push_back(T(y-N*N+2*mat_size, y+mat_size,-1));
        tripletList.push_back(T(y+mat_size,y+mat_size,-1.));
        tripletList.push_back(T(y+mat_size,y+mat_size,hat_U));
    }

    for (int y = N*N; y < mat_size; ++y) 
    {
        tripletList.push_back(T(y-N*N, y, 1));
        tripletList.push_back(T(y, y, -1));
        tripletList.push_back(T(y, y, hat_U));
        tripletList.push_back(T(y-N*N+mat_size, y+mat_size, 1));
        tripletList.push_back(T(y+mat_size, y+mat_size, -1));
        tripletList.push_back(T(y+mat_size, y+mat_size, hat_U));
    }
        
    delta_tau.setFromTriplets(tripletList.begin(), tripletList.end());
}

void get_m_matrix_same_4_all(const int Nt, const int N, const double hat_t, const double hat_U, SpMat& ret) 
{
    int mat_size = N*N*Nt; 
    SpMat delta_tau((mat_size << 1), (mat_size << 1));
    SpMat s1(mat_size, mat_size), s2(mat_size, mat_size); 

    get_m_matrix_tau_free(Nt, N, hat_t, s1, s2); 
    get_m_matrix_delta_tau(Nt, N, hat_U, delta_tau); 

    ret = delta_tau + Eigen::kroneckerProduct((Eigen::Matrix2d() << 0,0,1,0).finished(), s1) + Eigen::kroneckerProduct((Eigen::Matrix2d() << 0,1,0,0).finished(), s2); 
}

void get_m_matrix_xi(const double hat_u, Eigen::VectorXd& xi, SpMat& ret)
{
    ret = xi.asDiagonal() * sqrt(hat_u);
}
