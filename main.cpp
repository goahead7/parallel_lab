#include "src/types.h"

/* Compile time assert macro */
#define ASSERT_CONCAT_(a,b) a##b
#define ASSERT_CONCAT(a,b) ASSERT_CONCAT_(a,b)
#define ct_assert(e) enum {ASSERT_CONCAT(asssert_line_, __LINE__) = 1/(!!(e))}


experiment_result RunExperiment(integrate_function I)
{
    double t0 = omp_get_wtime();
    double res = I(-1, 1, Quadratic);
    double t1 = omp_get_wtime();

    experiment_result Result;
    Result.Result = res;
    Result.TimeMS = t1 - t0;

    return Result;
}

void ShowExperimentResult(integrate_function I)
{
    double T1;
    printf("%10s. %10s %10sms %10s\n", "Threads", "Result", "Time", "Acceleration");
    for(unsigned T = 1; T <=omp_get_num_procs(); T++)
    {
        experiment_result Experiment;
        set_num_threads(T);
        Experiment = RunExperiment(I);
        if (T == 1) {
            T1 = Experiment.TimeMS;
        }
        printf("%10d. %10g %10gms %10g\n", T, Experiment.Result, Experiment.TimeMS, T1/Experiment.TimeMS);
    }
    printf("\n");
}

int main() {

    set_num_threads(1);

    printf("Integrate_mutex_cpp\n");
    ShowExperimentResult(integrate_cpp_mtx);

    printf("Integrate_partial_sum\n");
    ShowExperimentResult(integrate_partial_sum);

    printf("Integrate_reduction_cpp\n");
    ShowExperimentResult(integrate_cpp_reduction);

    printf("Integrate_reduction_range_cpp\n");
    ShowExperimentResult(integrate_range_reduction);

    printf("Integrate_align_omp\n");
    ShowExperimentResult(integrate_align_omp);

    printf("Integrate_crit_omp\n");
    ShowExperimentResult(integrate_crit);

    printf("Integrate_false_sharing_omp\n");
    ShowExperimentResult(integrate_false_sharing);

    printf("Integrate_reduction_omp\n");
    ShowExperimentResult(integrate_reduction);

    return 0;
}
