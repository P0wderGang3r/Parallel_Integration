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

#define ndx 100

//#define CACHE_LINE std::hardware_destructive_interference_size
#define CACHE_LINE 64u

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
                for (unsigned j = 1; j <= (unsigned)thread::hardware_concurrency(); j++) {
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
}