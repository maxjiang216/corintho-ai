from multiprocessing import Pool
import numpy as np
import time


def f(n):
    return np.sqrt(n * 100)


nums = []
for i in range(1000000):
    nums.append(i)

start = time.time()
for i in nums:
    f(i)
print(f"Non-multiprocessing version took {time.time()-start} seconds")

for i in range(1, 9):
    pool = Pool(processes=i)
    start = time.time()
    for n in nums:
        f(n)
    print(f"{i}-process version took {time.time()-start} seconds")
    pool.close()
