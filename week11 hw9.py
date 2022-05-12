import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


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


img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

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

H, W = img.shape
X = np.zeros((int(H/8), int(W/8), 8, 8))    # 이미지를 8*8로 저장할 배열
# X_hat = np.zeros((H, W))

# 이미지를 8*8로 쪼개서 저장
for i in range(int(H/8)):
    for j in range(int(W/8)):
        X[i, j,:, :] = img[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8]

print("image 0 1 big")
print(X[0, 0])
print(X[0, 1])
print(X[50, 50])

# Y 에 DCT 저장
Y1 = np.zeros((int(H/8), int(W/8), 8, 8))
Y = np.zeros((int(H/8), int(W/8), 8, 8))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        Y1[i, j, :, :] = np.matmul(C, X[i, j])
        Y[i, j, :, :] = np.round(np.matmul(Y1[i, j], CT))

print("DCT 0, big")
print(Y[0, 0])
print(Y[50, 50])

Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]

# Y Quantization -> Y_hat
Y_hat = np.zeros((int(H/8), int(W/8), 8, 8))

for i in range(int(H/8)):
    for j in range(int(W/8)):
        Y_hat[i, j,:, :] = np.round(Y[i, j] / Q)

print("Q 0, big")
print(Y_hat[0, 0])
print(Y_hat[50, 50])

# Y_hat zigzag scan
zig_Y = np.zeros((int(H/8), int(W/8), 64))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        zig_Y[i, j] = zigzag(Y_hat[i, j])

print("zigzag 0, 1, big")
print(zig_Y[0, 0])
print(zig_Y[0, 1])
print(zig_Y[50, 50])

# Run length coding
run_level = list()
num_zero = 0
alert = 1

for i in range(int(H/8)):
    for j in range(int(W/8)):
        for k in zig_Y[i, j]:
            if alert == 1:
                alert = 0
                run_level.append(k)
            elif k != 0:
                run_level.append((num_zero, k))
                num_zero = 0
            else:
                num_zero += 1
        run_level.append("EOB")
        num_zero = 0
        alert = 1

print("run level ~15")
print(run_level[0:15])

# De run length coding
de_run_level = np.zeros((int(H/8), int(W/8), 64))

column = 0
row = 0
index = 0
alert = 1
# i[0] : 첫번째 0의 갯수, i[1] : 0 이후에 오는 숫자
for i in run_level:
    """
    print(i)
    print(column)
    print(row)
    print()
    """
    if alert == 1:   # 처음일 때
        de_run_level[column, row, index] = i    # DC값 설정
        index += 1  # 다음으로 넘어감
        alert = 0

    elif i == 'EOB':    # EOB 일 때
        index = 0   # 인덱스 초기화
        alert = 1
        if column == 63 and row == 63:    # 마지막이라면 종료
            break
        elif row == 63:  # 마지막 행일 때
            column += 1 # 열을 옮김
            row = 0     # 행 초기화
        else:   # 마지막 행이 아닐 때
            row += 1    # 행 옮김

    elif i[0] == 0:  # 뒤에 0이 없는 경우
        de_run_level[column, row, index] = i[1]  # 숫자를 설정
        index += 1  # 다음으로 넘어감

    else:  # 뒤에 0이 있는 경우
        num_zero = i[0]  # 0의 갯수를 num_zero에 저장
        while (num_zero > 0):  # num_zero만큼 반복
            de_run_level[column, row, index] = 0  # 0으로 설정
            num_zero -= 1  # 0의 갯수를 한개 줄임
            index += 1  # 다음으로 넘어감
        de_run_level[column, row, index] = i[1]  # 0 이후에 나오는 숫자를 설정
        index += 1  # 다음으로 넘어감

print("de_run_levels 0, 1, 2, big")
print(de_run_level[0, 0, :])
print(de_run_level[0, 1, :])
print(de_run_level[0, 2, :])
print(de_run_level[50, 50, :])

# inverse zigzag
inv_zigzag = np.zeros((int(H/8), int(W/8), 8, 8))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        inv_zigzag[i, j, :, :] = inverse_zigzag(de_run_level[i, j])

print("inv zigzag ~2 big")
print(inv_zigzag[0, 0])
print(inv_zigzag[50, 50])

# Dequantization
deq = np.zeros((int(H/8), int(W/8), 8, 8))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        deq[i, j, :, :] = inv_zigzag[i, j, :, :] * Q

print("Dequantization 0, 1, big")
print(deq[0, 0])
print(deq[0, 1])
print(deq[50, 50])

# idct
Cinv = np.linalg.inv(C)
CTinv = np.linalg.inv(CT)

CinvV_hat = np.zeros((int(H/8), int(W/8), 8, 8))
X_hat = np.zeros((int(H/8), int(W/8), 8, 8))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        CinvV_hat[i, j, :, :] = np.matmul(Cinv, deq[i, j])
        X_hat[i, j, :, :] = np.round(np.matmul(CinvV_hat[i, j], CTinv))

print("X_hat 0 1, big")
print(X_hat[0,0])
print(X_hat[0,1])
print(X_hat[50, 50])

# sum to image
img2 = np.zeros((H, W))
for i in range(int(H/8)):
    for j in range(int(W/8)):
        img2[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = X_hat[i, j, :, :]

plt.imshow(img2, cmap='gray')
plt.show()