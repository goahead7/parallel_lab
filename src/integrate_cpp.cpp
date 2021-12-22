#include "types.h"

double integrate_cpp_mtx(double a, double b, function f)
{
    using namespace std;

    unsigned int T = get_num_threads();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b-a)/STEPS;

    for(unsigned t = 0; t < T; t++)
    {
        threads.emplace_back([=, &result, &mtx]()
        {
            double R = 0;
            for(unsigned i = t; i < STEPS; i+=T)
                R += f(dx*i + a);
            {
                scoped_lock lck(mtx);
                result += R;
            }
        });
    }

    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

double integrate_partial_sum(double a, double b, function f)
{
    double result = 0, dx = (b-a)/STEPS;
    unsigned T = get_num_threads();
    auto vec = std::vector<partial_sum>(T, partial_sum{0.0});

    std::vector<std::thread> thread_vec;

    auto thread_procedure = [dx, T, f, a, &vec](auto t)
    {
        for(auto i = t; i < STEPS; i+=T)
            vec[t].value += f(dx*i + a);
    };

    for(unsigned t = 1; t < T; t++)
    {
        thread_vec.emplace_back(thread_procedure, t);
    }

    thread_procedure(0);
    for(auto &thread : thread_vec)
    {
        thread.join();
    }

    for(auto elem : vec)
    {
        result += elem.value;
    }

    return result*dx;
}

double integrate_cpp_reduction(double a, double b, function f)
{
    std::atomic<double> result = {0.0};
    double dx = (b - a) / STEPS;
    unsigned int num = get_num_threads();
    std::vector<std::thread> threads;

    auto fun = [dx, &result, f, a, num] (auto t)
    {
        double R = 0;
        for (unsigned i = t; i < STEPS; i+=num) {
            R = R + f(i*dx+a);
        }

        result = result + R;
    };

    threads.reserve(num);
    for (unsigned i = 1; i < num; ++i) {
        threads.emplace_back(fun, i);
    }

    fun(0);

    for (auto &thread:threads) {
        thread.join();
    }

    return result*dx;
}

double integrate_range_reduction(double a, double b, function f) {
    double dx = (b - a)/STEPS;
    return reduce_range(a, b, STEPS, f, [](auto x, auto y){return x + y;}, 0.0)*dx;
}