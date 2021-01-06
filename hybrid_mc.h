#ifndef HYBRID_MC_H
#define HYBRID_MC_H 

#include "m_matrix.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <cstdlib>
#include <EigenRand-0.3.1/EigenRand/EigenRand>

#include "helper_funcs.h"


class Trajectory {
public:
    Trajectory(const int Nt, const int N, const double hat_t, const double hat_u, const int max_epochs);
    const int N, Nt; 
    const double hat_t, hat_u; 

    Eigen::MatrixXd xis; // save all xi's 

    int max_epochs; // Total epochs 

    SpMat m_mat_indep; // independent of xi part of m_maitrx

    double time_step;

    void evolve(const double time_step, const int max_steps=5);
private: 
    /** 
     * Sampling phi
     */
    void _generate_phi();

    /** 
     * Calculate fermionic force
     */
    void _calculate_force();

    /**
     * Calculate Hamiltonian
     */
    double _calculate_hamiltonian();

    /**
     * Evolve with leap frog
     */
    void _leap_frog(int max_steps);

    int full_size, mat_size, lat_size; // Sizes relative to our model

    int rejected_steps; 
    SpMat m_mat; // full m_matrix used in solving fermionic force 
    Eigen::VectorXd phi1, phi2; // two component of phi 
    Eigen::VectorXd x1, x2, buf1, buf2;  // x's corresponding to phi

    Eigen::VectorXd xi; // curent state of xi
    Eigen::VectorXd pi; // current state of pi
    Eigen::VectorXd force; // current fermionic force
    Eigen::VectorXd hamiltonians; // all hamiltonian of states
    // Eigen::BiCGSTAB<SpMat> solver;
    Eigen::ConjugateGradient<SpMat> solver;

    SpMat f_mat;
    std::mt19937_64 urng; // random number generator
};

/* 
void my_input() { // Python style input function, useful for debug
    std::string buf;
    std::getline(std::cin, buf);
}
*/
#endif 