#include "OMP_realizations.cpp"

#define n 10000000
#define numOfTypes 5


namespace {
    
    //---------------���������� ���������-------------------

    //------------------>integrate_cpp_base
    double integrate_cpp_base(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;

        //�������� ���������� �������
        unsigned T = std::thread::hardware_concurrency();
        std::vector<double> results(T);

        //������, ���������� �������, ������� ����� ����� ��������� ������ �� �������
        auto thread_proc = [=, &results](unsigned t) {

            results[t] = 0;
            for (unsigned i = t; i < n; i += T)
                results[t] += f(dx * i + a);
        };

        //������� ������ ����������� �������
        std::vector<std::thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_proc, t);

        thread_proc(0);

        //�������������� ���������� ������� �� �������
        for (auto& thread : threads)
            thread.join();

        //���������� ���������� ���������� ������� �� ������� � ���� ����� ����������
        for (unsigned i = 0; i < T; ++i)
            res += results[i];

        return res * dx;
    }

    //------------------>integrate_cpp_cs
    double integrate_cpp_cs(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;

        //�������� ���������� �������
        unsigned T = std::thread::hardware_concurrency();

        std::mutex mtx;

        //������, ���������� �������, ������� ����� ����� ��������� ������ �� �������
        auto thread_process = [=, &res, &mtx](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < n; i += T)
                l_res += f((float)(dx * i + a));
            {
                //�� ���� ����� ��������� scoped_lock, �� � ���� ������������ std
                //� ���� �� �����������, ������ ������� �� ���������� ���
                std::lock_guard<std::mutex> lck(mtx);
                res += l_res;
                //��� ����������� ������� � ���������� � ������� mutex �� ���������
                //������ � ���������� �������, ����� ������� ����������, �� �������
            }
        };

        //������� ������ ����������� �������
        std::vector<std::thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //�������������� ���������� ������� �� �������
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }

    //------------------>integrate_cpp_atomic
    double integrate_cpp_atomic(double a, double b, f_t f) {
        double dx = (b - a) / n;
        std::atomic<double> res = 0;

        //�������� ���������� �������
        unsigned T = std::thread::hardware_concurrency();

        //������, ���������� �������, ������� ����� ����� ��������� ������ �� �������
        auto thread_process = [=, &res](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < n; i += T)
                l_res += f((float)(dx * i + a));

            //������ �������� �� ���������, ������� �� ������� �� ������ "atomic"
            res.store(res.load() + l_res);
            //��� ��� ���������� ��������, �� ������ ����� �������� �������������
            //���������� ������� ������� � ����������, ��� ������������ ����� �������
        };

        //������� ������ ����������� �������
        std::vector<std::thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //�������������� ���������� ������� �� �������
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }

    //------------------>integrate_cpp_reduce
    double integrate_cpp_reduce(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        return res * dx;
    }

    //------------------>integrate_cpp_mtx
    double integrate_cpp_mtx(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        return res * dx;
    }

    //----------------�������� �������������------------------

    //--------------------�������� �����-----------------------

    void cpp_start() {
        integrate_t integrate_type[numOfTypes];

        int typeNum = 0;

        integrate_type[typeNum++] = integrate_cpp_base;
        integrate_type[typeNum++] = integrate_cpp_cs;
        integrate_type[typeNum++] = integrate_cpp_atomic;
        integrate_type[typeNum++] = integrate_cpp_reduce;
        integrate_type[typeNum++] = integrate_cpp_mtx;

        std::cout << "CPP results" << std::endl;
        run_experiments(integrate_type, numOfTypes);
    }
}