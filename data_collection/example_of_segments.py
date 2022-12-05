
import warnings
import numpy as np

def measure(l:'list[list[int]]') -> float:
    x1 = []
    y1 = []
    z1 = []
    for p in l:
        x1.append(p[0])
        y1.append(p[1])
        z1.append(p[2])
    x = np.array(x1)
    y = np.array(y1)
    z = np.array(z1)
    return x.var() + y.var() + z.var() 
    pass

def measure_all(aloc:'list[int]', arr:'list[list[int]]') -> float:
    x = []
    sum = 0
    score = 0
    for len in aloc:
        x.clear()
        for i in range(len):
            x.append(arr[sum+i])
        score += measure(x)
        sum += len
    return score
#%%
# def dfs(numbers:'list', pos:int, count:int, n:int, ans:list, maxkind:int, nowv:list[float], All_points):
def dfs(numbers, pos, count, n, ans, maxkind, nowv, All_points):
    if pos >= maxkind - 1:
        numbers[pos] = n-count
        # print(numbers)
        if measure_all(numbers,All_points) < nowv:
            nowv[0] = measure_all(numbers,All_points)
            for i in range(maxkind):
                ans[i] = numbers[i]
        return

    for i in range(1, n-count-(maxkind-pos-1)+1):
        numbers[pos] = i
        dfs(numbers, pos+1, count+i, n, ans, maxkind, nowv, All_points)


#%%
All_points = [[0, 0, 0 , 0, 0, 0, 0], [0, 0, 0.05 , 0, 0, 0, 0], [0, 0, 0.1 , 0, 0, 0, 0], [0, 0, 0.1 , 0, 0, 0, 1]]
max_kind = 3

numbers = [0 for _ in range(max_kind)]
nowv = [2147483647] #big number
ans = [0 for _ in range(max_kind)]

dfs(numbers,0,0,All_points.__len__(),ans,max_kind,nowv,All_points)

sum = 0
for i in range(ans.__len__()):
    len = ans[i]
    print('Kind {}: '.format(i+1),end='')
    for pi in range(len):
        print(All_points[sum+pi],end=' ')
    print()
    sum += len
pass

