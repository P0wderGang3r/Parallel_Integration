#include <iostream>
#include <omp.h>
#include <time.h>

namespace {

    typedef double (*f_t) (double);

    #define n 10000000

    double g(double x) {
        return x * x;
    }

    //---------------Вычисление интеграла-------------------

    //------------------>integrate_seq
    double integrate_seq(double a, double b, f_t f) {

        double dx = (b - a) / n;
        double res = 0;

        for (int i = 0; i < n; ++i) {
            res += f(dx * i + a);
        }
        return res * dx;
    }


    //------------------>integrate_omp_base
    double integrate_omp_base(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        unsigned T;
        double* results;

#pragma omp parallel shared(results, T)
        {
            unsigned t = (unsigned)omp_get_thread_num();

#pragma omp single
            {
                T = (unsigned)omp_get_num_threads();

                results = new double[T];
                for (int i = 0; i < T; i++)
                    results[i] = 0;

                if (!results)
                    abort();
            } //барьерная реализация

            for (size_t i = t; i < n; i += T) {
                results[t] += f(dx * i + a);
            }
        }
        for (size_t i = 0; i < T; i++)
            res += results[i];

        delete results;
        return res * dx;
    }


    //------------------>integrate_omp_cs
    double integrate_omp_cs(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / n;
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();

            for (size_t i = t; i < n; i += T)
            {
#pragma omp critical
                {
                    res += f(dx * i + a);
                }
            }
        }

        return res * dx;
    }


    //------------------>integrate_omp_atomic
    double integrate_omp_atomic(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();

            for (size_t i = t; i < n; i += T)
            {
                double val = f((double)(dx * i + a));
#pragma omp atomic 
                res += val;
            }
        }

        return (double)(res * dx);
    }


    //------------------>integrate_omp_for
    double integrate_omp_for(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        int i;

#pragma omp parallel for shared (res) 
        for (i = 0; i < n; ++i)
        {
            double val = f((double)(dx * i + a));
#pragma omp atomic 
            res += val;
        }

        return (double)(res * dx);
    }


    //------------------>integrate_omp_reduce
    double integrate_omp_reduce(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        int i;

#pragma omp parallel for reduction(+: res) 
        for (i = 0; i < n; ++i)
            res += f((double)(dx * i + a));
        return (double)(res * dx);
    }

    //----------------Создание экспериментов------------------

    struct experiment_result_t
    {
        double result;
        double time;
    };

    typedef double (*integrate_t)(double a, double b, f_t f);

    experiment_result_t run_experiment_omp(integrate_t integrate) {
        experiment_result_t result;
        double t0 = omp_get_wtime();
        result.result = integrate(-1, 1, g);
        result.time = omp_get_wtime() - t0;
        return result;
    }

    void run_experiments_omp(integrate_t* integrate_type) {
        experiment_result_t r;
        for (int i = 0; i < 6; i++) {
            r = run_experiment_omp(integrate_type[i]);
            std::cout << "result: " << r.result << " " << r.time << std::endl;
        }
    }

    //--------------------Основной поток-----------------------

    void omp_start() {
        integrate_t integrate_type[6];

        integrate_type[0] = integrate_seq;
        integrate_type[1] = integrate_omp_base;
        integrate_type[2] = integrate_omp_cs;
        integrate_type[3] = integrate_omp_atomic;
        integrate_type[4] = integrate_omp_for;
        integrate_type[5] = integrate_omp_reduce;

        run_experiments_omp(integrate_type);
    }
}