#include "shared_within_realizations.cpp"

#define ndx 10000000

namespace {//---------------Вычисление интеграла-------------------

    //------------------>integrate_seq
    //Последовательный алгоритм численного интегрирования
    double integrate_seq(double a, double b, f_t f) {

        double dx = (b - a) / ndx;
        double res = 0;

        for (int i = 0; i < ndx; ++i) {
            res += f(dx * i + a);
        }
        return res * dx;
    }


    void seq_start() {
        std::cout << "SEQ results" << std::endl;

        double t0 = omp_get_wtime();
        double integrate_seq_res = integrate_seq(-1, 1, g);
        std::cout << "result: " << integrate_seq_res << " " << omp_get_wtime() - t0 << std::endl;
    }
}