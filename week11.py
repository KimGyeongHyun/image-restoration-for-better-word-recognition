
import numpy as np
import math

def zigzag(input):
    h, v, i = 0,0,0
    output = np.zeros((64))
    while ((v < 8) and (h < 8)): # going up
        if ((h + v) % 2) == 0:
            if (v == 0):
                output[i] = input[v, h]
                if (h == 8):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == 7) and (v < 7)):
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > 0) and (h < 7)):
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if ((v == 7) and (h <= 7)):
                output[i] = input[v, h]
                h = h + 1
                i = i + 1
            elif (h == 0):
                output[i] = input[v, h]
                if (v == 7):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < 7) and (h > 0)):
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == 7) and (h == 7)):
            output[i] = input[v, h]
            break
    return output

def inverse_zigzag(input):
    h, v, i = 0,0,0
    output = np.zeros((8,8))
    while ((v < 8) and (h < 8)):
        if ((h + v) % 2) == 0: # going up
            if (v == 0):
                output[v, h] = input[i]
                if (h == 8):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == 7) and (v < 7)):
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > 0) and (h < 7)):
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:  # going down
            if ((v == 7) and (h <= 7)):
                output[v, h] = input[i]
                h = h + 1
                i = i + 1
            elif (h == 0):
                output[v, h] = input[i]
                if (v == 7):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < 7) and (h > 0)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == 7) and (h == 7)):  # bottom right element
            output[v, h] = input[i]
            break
    return output

N = 8
C = []
for i in range(N):
    row = []
    for j in range(N):
        if i == 0:
            P = 1
        else:
            P = math.sqrt(2)

        cij = P / math.sqrt(N) * math.cos(math.pi / N * (j + 1/2) * i)
        row.append(cij)
    C.append(row)

C = np.array(C)
CT = np.transpose(C)
print(C)


X = np.random.randint(0,255,size=(8, 8))
print(X)

Y1 = np.matmul(C, X)    # CX
Y = np.round(np.matmul(Y1, CT)) # CX * CT
print(Y)



Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]

Y_hat = np.round(Y / Q)
print(Y)

zig_Y = zigzag(Y_hat)
print(zig_Y)


run_level = list()
num_zero = 0

for i in zig_Y:
    if len(run_level) == 0:
        run_level.append(i)
    elif i != 0:
        run_level.append((num_zero, i))
        num_zero = 0
    else:
        num_zero += 1

run_level.append("EOB")
print(run_level)

de_run_level = np.zeros(64)
index = 0
for i in run_level:
    if not np.any(de_run_level):
        de_run_level[index] = i
        index += 1

    elif i == 'EOB':
        break
    elif i[0] == 0:
        de_run_level[index] = i[1]
        index += 1
    else:
        num_zero = i[0]
        while(num_zero > 0):
            de_run_level[index] = 0
            num_zero -= 1
            index += 1
        de_run_level[index] = i[1]
        index += 1

print(de_run_level)

inv_zigzag = inverse_zigzag(de_run_level)
print(inv_zigzag)


# 겹치는 부분 없도록 구현
# 코드 안에서 달라지는 이유 찾기