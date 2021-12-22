#include "types.h"

double integrate_align_omp(double a, double b, function f)
{
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    partial_sum* accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            accum = (partial_sum*) calloc(CACHE_LINE, T * sizeof(partial_sum));
        }

        for (unsigned i = t; i < STEPS; i += T)
        {
            accum[t].value += f(dx*i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i)
    {
        result += accum[i].value;
    }

    free(accum);

    return result * dx;
}

double integrate_crit(double a, double b, function f)
{
    double result = 0, dx = (b - a)/STEPS;

#pragma omp parallel
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();

        for (unsigned i = t; i < STEPS; i += T)
        {
            R += f(i * dx + a);
        }
#pragma omp critical
        result += R;
    }

    return result * dx;
}

double integrate_false_sharing(double a, double b, function f)
{
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    double* accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (double*) calloc(T, sizeof(double));
        }

        for (unsigned i = t; i < STEPS; i += T)
        {
            accum[t] += f(dx*i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i];
    }

    return result * dx;
}

double integrate_reduction(double a, double b, function f)
{
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}