import timeit


def time_test(i):
    l = [x*x for x in range(0, i)]
    return l


def measure_time():

    print("the module name is ", __name__)

    exec_t = timeit.timeit("time_test(100)", setup="from test_module import time_test", number=100)
    print("the execution time of the function is ", str(exec_t))
