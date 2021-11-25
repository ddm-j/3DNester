# importing the required module
import timeit

# code snippet to be executed only once
mysetup_1 = "import numpy as np\n" \
             "myarr = np.ones((5000,5000))\n" \
            "ind = np.array([1, 2])"

mysetup_2 = "import numpy as np\n" \
             "myarr = np.ones((5000,5000))\n" \
            "ind = np.array([1, 2])"

# code snippet whose execution time is to be measured
mycode_np = '''
def example():
    myarr[ind[0],ind[1]] = 0
'''

mycode_cy = '''
def example():
    marr[tuple(ind)] = 0
'''

# timeit statement
print(timeit.timeit(setup=mysetup_1,
                    stmt=mycode_np,
                    number=10000000))

print(timeit.timeit(setup=mysetup_1,
                    stmt=mycode_cy,
                    number=10000000))