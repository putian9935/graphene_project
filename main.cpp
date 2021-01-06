#define EIGEN_USE_MKL_ALL
#include <iostream> 
#include "m_matrix.h"
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/Core>
#include <chrono> 
#include "hybrid_mc.h"
#include <ctime>

void testSolvers()
{
    constexpr auto N = 3, Nt = 10;
    constexpr auto hat_t = 2e-2, hat_U = 2e-4; 
    SpMat m_mat(N*N*Nt*2, 2*N*N*Nt); 
    SpMat f_mat(N*N*Nt*2, 2*N*N*Nt);
    get_m_matrix_same_4_all(Nt, N, hat_t, hat_U, m_mat); 

    Eigen::VectorXd x, y;
    Eigen::BiCGSTAB<SpMat> solver;
    // Eigen::ConjugateGradient<SpMat> solver;
    Eigen::LeastSquaresConjugateGradient<SpMat> solverb; 
    solver.setTolerance(1e-5);
    auto t1 = std::chrono::high_resolution_clock::now();

    
    

    for (int epoch = 0; epoch < 1000; ++ epoch)
    {
        // f_mat = (m_mat * m_mat.transpose()).eval();
        //x = solver.compute(m_mat).solve(Eigen::VectorXd::Random(N*N*Nt*2));
        x = solver.compute(m_mat).solve(Eigen::VectorXd::Random(N*N*Nt*2));
        y = solver.compute(m_mat.transpose()).solve(x);
        // std::cout << "#iterations:     " << solverb.iterations() << std::endl;
        //std::cout << "estimated error: " << solverb.error()      << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << duration << '\n';
    //std::cout << x << '\n';
}

void test_diagonal()
{
    constexpr auto N =3, Nt = 2;
    constexpr auto hat_t = 1e-2, hat_U = 1e-3; 
    SpMat m_mat(N*N*Nt*2, 2*N*N*Nt); 

    Eigen::VectorXd xi = Eigen::VectorXd().Random(N*N*Nt*2); 

    get_m_matrix_xi(hat_U, xi, m_mat);
    std:: cout << m_mat << '\n';
}

void test_cwiseproduct()
{
    Eigen::Vector3d x(1,2,3);
    Eigen::Vector3d y(1,3,5);
    auto z = x.cwiseProduct(y);
    std::cout << x << std:: endl << z << std::endl;
}

void test_EigenRand()
{
    std::mt19937_64 urng{ 42 };
    auto mat = Eigen::Rand::normal<Eigen::VectorXf>(4, 1, urng).cast<double> ();
    std::cout << mat.eval() << '\n';
}

void test_evolve()
{
    constexpr int N = 3, Nt = 10;
    constexpr double hat_t = 2e-2, hat_u = 2e-4; 
    Trajectory traj(Nt, N, hat_t, hat_u, 2000);
    traj.evolve(.3,5);
}

int main() 
{
    Eigen::setNbThreads(2);
    int n = Eigen::nbThreads( );
    std:: cout << "Running on " << n << " thread(s).\n";

    
    int tt = clock();
    // testSolvers();
    // test_diagonal();
    // test_cwiseproduct();
    test_evolve();

    // test_EigenRand();
    std:: cout << "Time elapsed: " << (clock() - tt) / 1e6 << "s.\n";
    return 0;
}