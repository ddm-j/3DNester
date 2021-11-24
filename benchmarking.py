# importing the required module
import timeit

# code snippet to be executed only once
mysetup_np = "import numpy as np"

# code snippet whose execution time is to be measured
mycode_np = '''
def example():
    myarr = np.zeros(5000)
    for x in range(1000):
        myarr[np.random.randint(1,4999)] = 1
'''

mycode_py = '''
def example():
    mylist = []
    for x in range(1000):
        mylist.append[1]
    mylist = np.array(mylist)
'''

# timeit statement
print(timeit.timeit(setup=mysetup_np,
                    stmt=mycode_np,
                    number=10000))

print(timeit.timeit(setup=mysetup_np,
                    stmt=mycode_py,
                    number=10000))