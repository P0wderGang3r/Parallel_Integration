#include "CPP_realizations.cpp"

using namespace std;

int main()
{
	//freopen("input.txt", "w", stdin);
	//freopen("output.txt", "w", stdout);
	//cout << std::thread::hardware_concurrency();
	seq_start(); //seq
	omp_start(); //base, cs, atomic, for, reduce, mtx
	cpp_start(); //base, cs, atomic, reduce, mtx
	//OMP_reduce_is_so_unique(integrate_omp_reduce);
	//CPP_reduce_is_so_unique(integrate_cpp_reduce);
}