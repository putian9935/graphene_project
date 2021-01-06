#ifndef HELPER_FUNCS_H_INCLUDED
#define HELPER_FUNCS_H_INCLUDED 

#include <iostream> 
#include <cstdlib> 

void input();
void prepare_for_percentage_readout(const char * const prefix);
void update_percentage_readout(int percentage, const char * const prefix);
void finish_percentage_readout();
#endif 