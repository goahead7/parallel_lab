#include "types.h"

unsigned get_num_threads()
{
    return num_treads;
}

void set_num_threads(unsigned t)
{
    num_treads = t;
    omp_set_num_threads(t);
}