#include "helper_funcs.h"

void input() { // Python style input function, useful for debug
    std::string buf;
    std::getline(std::cin, buf);
}
void prepare_for_percentage_readout(const char * const prefix)
{
    fprintf(stderr, "%s: 0%%", prefix);
}

void update_percentage_readout(int percentage, const char * const prefix)
{
    fprintf(stderr, "\r%s: %d%%", prefix, percentage);
    fflush(stderr);
}

void finish_percentage_readout()
{
    fprintf(stderr, "\n");
}