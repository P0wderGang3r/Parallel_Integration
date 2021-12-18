#include <iostream>
#include <time.h>
#include <omp.h>

#include <mutex>
#include <atomic> 
#include <thread>

#include <vector>
#include <numeric>

#include <Windows.h>
#include <string>


#define ndx 10000000

#define numOfOMPTypes 6 //With reduce equals 6
#define numOfCPPTypes 4 //With reduce equals 5

namespace {
    using std::thread;

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

    void run_experiments(integrate_t* integrate_type, std::string* typeNames, int numOfTypes) {
        experiment_result_t r;
        for (int i = 0; i < numOfTypes; i++) {
            try {
                std::cout << "Integrate type name: " << typeNames[i] << std::endl;
                for (unsigned j = 1; j <= thread::hardware_concurrency(); j++) {
                    numOfThreads = j;
                    r = run_experiment(integrate_type[i]);
                    std::cout << "" << r.result << " " << r.time << std::endl;
                }
                //std::cout << "result: " << r.result << " " << r.time << std::endl;

            }
            catch (...) {

            }
        }
    }

    unsigned reduceThreads = 1;

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