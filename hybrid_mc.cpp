#include "hybrid_mc.h"

Trajectory::Trajectory(const int Nt, const int N, const double hat_t, const double hat_u, const int max_epochs) : Nt(Nt), N(N), hat_t(hat_t), hat_u(hat_u), max_epochs(max_epochs)
{
    // Initilize a few sizes
    full_size = 2 * N * N * Nt;
    mat_size = N * N * Nt;
    lat_size = N * N;
    rejected_steps = 0;

    urng = std::mt19937_64();
    
    // xi = Eigen::VectorXd().Random(full_size);
    xi = Eigen::Rand::normal<Eigen::VectorXf>(full_size, 1, urng).cast<double>();
    get_m_matrix_same_4_all(Nt, N, hat_t, hat_u, m_mat_indep); // initialize
    // get_m_matrix_same_4_all(Nt, N, hat_t, hat_u, m_mat);
    solver.setTolerance(1e-5);
    force.conservativeResize(full_size);
}

void Trajectory::_generate_phi()
{
    get_m_matrix_xi(hat_u, xi, m_mat);
    m_mat += m_mat_indep;

    phi1 = sqrt(.5) * m_mat * Eigen::Rand::normal<Eigen::VectorXf>(full_size, 1, urng).cast<double> ();
    phi2 = sqrt(.5) * m_mat * Eigen::Rand::normal<Eigen::VectorXf>(full_size, 1, urng).cast<double> ();


    solver.compute(m_mat*m_mat.transpose());
    x1 = solver.solve(phi1);
    x2 = solver.solve(phi2);
    

    
    /*
    solver.compute(m_mat);
    buf1 = solver.solve(phi1);
    buf2 = solver.solve(phi2);

    solver.compute(m_mat.transpose());
    x1 = solver.solve(buf1);
    x2 = solver.solve(buf2);
    */
    
}

void Trajectory::_calculate_force()
{
    get_m_matrix_xi(hat_u, xi, m_mat);
    m_mat += m_mat_indep;

    
    solver.compute(m_mat*m_mat.transpose());
    x1 = solver.solve(phi1);
    x2 = solver.solve(phi2);

    force =-xi +(2. * sqrt(hat_u)) * x1.cwiseProduct(m_mat.transpose() * x1) + (2. * sqrt(hat_u)) * x2.cwiseProduct(m_mat.transpose() * x2);

    // force += (2. * sqrt(hat_u)) * x1.cwiseProduct(m_mat.transpose() * x1);
    // force += (2. * sqrt(hat_u)) * x2.cwiseProduct(m_mat.transpose() * x2);

    /*
    solver.compute(m_mat);
    buf1 = solver.solve(phi1);
    buf2 = solver.solve(phi2);

    solver.compute(m_mat.transpose());
    x1 = solver.solve(buf1);
    x2 = solver.solve(buf2);

    force = -xi +(2. * sqrt(hat_u)) * x1.cwiseProduct(buf1) + (2. * sqrt(hat_u)) * x2.cwiseProduct(buf2);
    */
    
}

double Trajectory::_calculate_hamiltonian()
{
    return .5 * (pi.squaredNorm() + xi.squaredNorm()) + phi1.dot(x1) + phi2.dot(x2);
}

void Trajectory::_leap_frog(int max_steps)
{
    _calculate_force();
    pi += (time_step / 2.) * force;
    for (int i = 0; i < (max_steps - 1); ++i)
    {
        xi += time_step * pi;
        _calculate_force();
        pi += time_step * force;
    }
    xi += time_step * pi;
    _calculate_force();
    pi += (time_step / 2.) * force;
}

void Trajectory::evolve(const double time_step, const int max_steps)
{
    this->time_step = time_step; // hhh
    xis = xi; // Initialize xis
    hamiltonians.conservativeResize(max_epochs);

    int percentage = 0; 
    const char * const prefix = "Evolving"; 
    prepare_for_percentage_readout(prefix);

    for(int epoch = 0; epoch < max_epochs; ++epoch){
        if ((epoch + 1) * 100 / max_epochs > percentage) 
        {
            percentage = (epoch + 1) * 100 / max_epochs; 
            update_percentage_readout(percentage, prefix);
        }
        
        Eigen::VectorXd prev_xi = xi; // make a copy 
        _generate_phi();

        // pi = Eigen::VectorXd().Random(full_size);
        pi = Eigen::Rand::normal<Eigen::VectorXf>(full_size, 1, urng).cast<double> ();
        
        double h_start = _calculate_hamiltonian();
        // std:: cout << h_start << '\n';
        _leap_frog(max_steps);
        double h_end = _calculate_hamiltonian();

        
        /* 
        std:: cout << h_end << ' ' << h_start << ' ' << exp(-h_end + h_start) << '\n';
        std::string buf;
        std::getline(std::cin, buf);
        */

        hamiltonians(epoch) = h_end - h_start;
        if(h_end < h_start) {
            xis.conservativeResize(xis.rows(), xis.cols()+1);
            Eigen::VectorXd xi_copy = xi;
            xis.col(xis.cols()-1) = xi_copy;
            continue;
        }
        if ((double) std::rand()/ (double) RAND_MAX > exp(-h_end + h_start)){
            xi = prev_xi;
            ++rejected_steps;
        }
        Eigen::VectorXd xi_copy = xi;
        xis.conservativeResize(xis.rows(), xis.cols()+1);
        xis.col(xis.cols()-1) = xi_copy;
           
    }

    finish_percentage_readout();
    std::cout << "Acceptance rate: " << (int) (100 * (1. - (double)rejected_steps /(double) max_epochs)) << "\%; "; 
    std::cout << " Acc./Tot.: " << (max_epochs - rejected_steps) << '/' << max_epochs << '\n';  
    std::cout << "Mean change in hamiltonian is " << hamiltonians.mean() << '\n';
}
