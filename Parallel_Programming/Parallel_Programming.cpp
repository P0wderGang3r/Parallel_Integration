#include "CPP_realizations.cpp"

using namespace std;

int main()
{
	seq_start(); //seq
	omp_start(); //base, cs, atomic, for, reduce, ?mtx?
	cpp_start(); //base, cs, atomic, reduce, ?mtx?
}