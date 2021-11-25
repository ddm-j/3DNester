# importing the required module
import timeit

# code snippet to be executed only once
mysetup_np = "import numpy as np\n" \
             "myarr = np.zeros((5000,5000))\n" \
             "index = 1000"
mysetup_cy = "import numpy as np\n" \
             "import utility_cy as uc\n" \
             "myarr = np.zeros((5000,5000))\n" \
             "index = 1000"

# code snippet whose execution time is to be measured
mycode_np = '''
def example():
    index = 0

    myarr[index, :] = 0
    myarr[:, index] = 0
    # Shift the rows/columns bigger than index "up one row & over (left) one column"
    myarr[i:-index, i:-index] = myarr[i + index:, i + index:]
    # Set last column = 0
    myarr[:, -1] = 0
'''

mycode_cy = '''
def example():
    index = 0
    arr = uc.remove_from_collision_array(index, myarr)
'''

# timeit statement
print(timeit.timeit(setup=mysetup_np,
                    stmt=mycode_np,
                    number=10000000))

print(timeit.timeit(setup=mysetup_cy,
                    stmt=mycode_cy,
                    number=10000000))