#include <stdio.h>
#include <assert.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <avx512fintrin.h>

using namespace std;

//AVX
double integrate(double a, double b, unsigned n)
{
    double dx = (b - a) / n;
    double result = 0;

    __m128d X = _mm_set_pd(a + dx, a);
    __m128d DX = _mm_set_pd1(2*dx);
    __m128d sum = _mm_set_pd1(0);

    unsigned Stride = sizeof(__m128d)/sizeof(double);
    assert(n >= Stride);
    assert(n % Stride == 0);

    for(unsigned int i = 0; i < n; i += Stride)
    {
        sum = _mm_add_pd(sum, _mm_mul_pd(X,X));
        X = _mm_add_pd(X, DX);
    }

    result = _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));

    result *= dx;
    return result;
}


void add_mtrx(const double* a, const double* b, std::size_t c, std::size_t r, double* result) {

    auto elementsInRegister = sizeof(__m256d)/sizeof(double);
    auto iterat = c*r/elementsInRegister;

    for(auto i = 0; i < iterat; i++)
    {
        __m256d aTerm = _mm256_load_pd(a);
        __m256d bTerm = _mm256_load_pd(b);
        __m256d sum = _mm256_add_pd(aTerm, bTerm);
        _mm256_storeu_pd(result, sum);

        a += elementsInRegister;
        b += elementsInRegister;
        result += elementsInRegister;
    }
}

//для AVX 256

void mul_mtrx(const double* A, const double* B, std::size_t cA, std::size_t rA,std::size_t cB, double* result)
{
    if (rA & 3)
        exit(-1);
    auto rB = cA;

    for (auto j = 0; j < rA; j += 4)
        {
            for (auto k = 0; k < cB; k++)
            {
                auto tempRes = _mm256_setzero_pd();

            for (auto i = 0; i < cA; i++) {

                     auto tempA  = _mm256_loadu_pd(&A[i*rA+j]);
                     auto tempB = _mm256_set1_pd(B[k*rB+i]) ;
                     tempRes  = _mm256_add_pd(_mm256_mul_pd(tempA,tempB),tempRes);
                  }
              _mm256_storeu_pd(&result[k*rA+j],tempRes);
        }

    }

}

void addMatrix(const double* a, const double* b, std::size_t c, std::size_t r, double* result) {
    auto elementsInRegister = sizeof(__m128d)/sizeof(double);
    auto iterations = c*r/elementsInRegister;

    for(auto i = 0; i < iterations; i++)
    {
        __m128d aTerm = _mm_load_pd(a);
        __m128d bTerm = _mm_load_pd(b);
        __m128d sum = _mm_add_pd(aTerm, bTerm);
        _mm_storeu_pd(result, sum);

        a += elementsInRegister;
        b += elementsInRegister;
        result += elementsInRegister;
    }
}

int main()
{
    double  intRes = integrate(-1, 1, 1000000);
    printf("Integrate Result: %f\n", intRes);

    uint32_t c = 4;
    uint32_t r = 4;

    auto* res_sum = (double*) malloc(c*r* sizeof(double));
    double A1[] = {1,2,3,4};
    double B1[] = {1,3,2,4};

    addMatrix(A1, B1, c, r, res_sum);
    printf("Sum matrix\n");
    cout<< res_sum[0] << ' '<< res_sum[1] << '\n'<< res_sum[2] << ' '<< res_sum[3]<<endl;

    double a[] = {1,2,3,4,5,6,7,8};
    double b[] = {1, 3, 2, 4};
    double res[8]={};

    mul_mtrx(a, b, 2, 4, 2, res);
    printf("Mul matrix\n");
    printf("");
    std::cout<<res[0]<<" "<<res[1]<<" "<<res[2]<<" "<<res[3]<<'\n' <<res[4]<<" "<<res[5]<<" "<<res[6]<<" "<<res[7];
    return 0;


}
