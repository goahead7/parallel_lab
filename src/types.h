#ifndef PARALLEL_FINAL_TYPES_H
#define PARALLEL_FINAL_TYPES_H

#include <iostream>
#include <omp.h>
#include <vector>
#include <thread>
#include <cstdlib>

#ifdef __cpp_lib_hardware_interface_size
using std::hardware_constructive_interface_size;
    using std::hardware_destructive_interface_size;
#else
constexpr std::size_t hardware_constructive_interface_size = 64u;
constexpr std::size_t hardware_destructive_interface_size = 64u;
#endif

#define STEPS 100000000
#define CACHE_LINE 64u

typedef double (*function)(double);

typedef double (*integrate_function) (double, double, function);

std::size_t ceil_div(std::size_t x, std::size_t y);

struct partial_sum
{
    alignas(64) double value;
};

struct experiment_result
{
    double Result;
    double TimeMS;
};

double Identity(double x);
double Linear(double x);
double Quadratic(double x);

static unsigned num_treads = std::thread::hardware_concurrency();

unsigned get_num_threads();

void set_num_threads(unsigned t);

double integrate_cpp_mtx(double a, double b, function f);
double integrate_partial_sum(double a, double b, function f);
double integrate_cpp_reduction(double a, double b, function f);
double integrate_range_reduction(double a, double b, function f);

double integrate_align_omp(double a, double b, function f);
double integrate_crit(double a, double b, function f);
double integrate_false_sharing(double a, double b, function f);
double integrate_reduction(double a, double b, function f);

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero);

template <class ElementType, class UnaryFn, class BinaryFn>
ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interface_size) ElementType value;

        explicit reduction_partial_result_t(ElementType value)
        {
            value = value;
        }
    };
    static auto reduction_partial_results =
            std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{zero});

    constexpr std::size_t k = 2;
    auto thread_proc = [=](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        std::size_t Mt = K / T;
        std::size_t it1;

        if(t < (K % T))
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = (K % T) * Mt + t;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
        {
            accum = reduce_2(accum, get(a + i*dx));
        }

        reduction_partial_results[t].value = accum;
    };


//    auto thread_proc_2 = [=](unsigned t)
//    {
//        constexpr std::size_t k = 2;
//        std::size_t s = 1;
//        while((t % (s * k)) && (t + s < T))
//        {
//            reduction_partial_results[t].value = f(reduction_partial_results[t].value,
//                                                   reduction_partial_results[t + s].value);
//            s *= k;
//            // ------------barrier------------ //
//        }
//    };


    auto thread_proc_2_ = [=](unsigned t, std::size_t s)
    {
        if(((t % (s * k)) == 0) && (t + s < T))
        {
            reduction_partial_results[t].value =
                    reduce_2(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
    {
        threads.emplace_back(thread_proc, t);
    }
    thread_proc(0);
    for(auto& thread : threads)
    {
        thread.join();
    }

    std::size_t s = 1;
    while(s < T)
    {
        for(unsigned t = 1; t < T; t++)
        {
            threads[t-1] = std::thread(thread_proc_2_, t, s);
        }
        thread_proc_2_(0, s);
        s *= k;

        for(auto& thread : threads)
        {
            thread.join();
        }
    }


//    for(unsigned t = 1; t < T; t++)
//    {
//        threads[t-1] = std::thread(thread_proc_2, t);
//    }
//    thread_proc_2(0);
//    for(auto& thread : threads)
//    {
//        thread.join();
//    }

    return reduction_partial_results[0].value;
}

#endif //PARALLEL_FINAL_TYPES_H
