#include <iostream>
#include <time.h>
#include <omp.h>

#include <mutex>
#include <atomic> 
#include <thread>

#include <vector>
#include <numeric>
#include <type_traits>

#include <Windows.h>
#include <string>

#define ndx 10000000

#define is_custom_barrier false
#if is_custom_barrier
#define reduce_type reduce_par_custom_barrier(reduction_buffer.data(), reduction_buffer.size(), [f, zero](const element_t& x, const element_t& y) {return element_t{ f(x.value, y.value) }; }, element_t{ zero }).value
#else
#define reduce_type reduce_par(reduction_buffer.data(), reduction_buffer.size(), [f, zero](const element_t& x, const element_t& y) {return element_t{ f(x.value, y.value) }; }, element_t{ zero }).value
#endif

namespace {
    using std::vector;
    using std::thread;

    unsigned max_threads = (unsigned)omp_get_max_threads();

    unsigned numOfThreads = 1;

    typedef double (*f_t) (double);

    double g(double x) {
        return x * x;
    }

    //----------------Создание экспериментов------------------


    struct experiment_result_t
    {
        double result;
        double time;
    };

    typedef double (*integrate_t)(double a, double b, f_t f);

    struct vectorType {
        integrate_t integrate;
        std::string typeName;
    };

    experiment_result_t run_experiment(integrate_t integrate) {
        experiment_result_t result;
        try {
            double t0 = omp_get_wtime();
            result.result = integrate(-1, 1, g);
            result.time = omp_get_wtime() - t0;
            return result;
        }
        catch (...) {
            result.result = 0;
            result.time = 0;
            return result;
        }
    }

    void run_experiments(std::vector<vectorType>* types) {
        experiment_result_t r;
        for (int i = 0; i < types->size(); i++) {
            try {
                std::cout << "Integrate type name: " << types->at(i).typeName << std::endl;
                for (unsigned j = 1; j <= thread::hardware_concurrency(); j++) {
                    numOfThreads = j;
                    r = run_experiment(types->at(i).integrate);
                    std::cout << "" << r.result << " " << r.time << std::endl;
                }
                //std::cout << "result: " << r.result << " " << r.time << std::endl;

            }
            catch (...) {

            }
        }
    }

    unsigned reduceThreads = 8;

    void OMP_reduce_is_so_unique(integrate_t integrate_type) {
        experiment_result_t r;
        try {
            std::cout << "Integrate type name: " << "OMP_reduce" << std::endl;
            numOfThreads = reduceThreads;
            r = run_experiment(integrate_type);
            std::cout << "result: " << r.result << " " << r.time << std::endl;

        }
        catch (...) {

        }
    }


    void CPP_reduce_is_so_unique(integrate_t integrate_type) {
        experiment_result_t r;
        try {
            std::cout << "Integrate type name: " << "CPP_reduce" << std::endl;
            numOfThreads = reduceThreads;
            r = run_experiment(integrate_type);
            std::cout << "result: " << r.result << " " << r.time << std::endl;

        }
        catch (...) {

        }
    }
}