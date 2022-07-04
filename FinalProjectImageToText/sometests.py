import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import math

a = 0
TB = 3


def gaussian_blur_func(kernel_size, sigma):

    extend_value = kernel_size//2
    return_filter = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            up = - ((i - extend_value) ** 2 + (j - extend_value) ** 2) / (2 * (sigma ** 2))
            exp = np.exp(up)
            exp = exp / (2 * math.pi * (sigma ** 2))
            return_filter[i, j] = exp

    return return_filter

def GaussianFilter(img, gaussian_img):
    H, W = img.shape
    gh, gw = gaussian_img.shape
    pad_img = np.zeros((H+ 2*(gh//2), W + 2*(gw//2)))
    pad_img[gh//2:H+gh//2, gw//2:W+gw//2] = img

    #필터링 후 이미지를 저장할 배열 선언
    filtered_img = np.zeros(img.shape)

    #입력 이미지와 가우시안 커널을 곱한 뒤, 이에 대한 결과값을 출력 이미지에 저장
    #해당 부분 잘 구현했으면 1점, 내장함수(cv2.filter2D()) 사용시 1점 감점

    for i in range(H):
        for j in range(W):
            filtered_img[i, j] = np.sum(pad_img[i:i+gh, j:j+gw] * gaussian_img)

    return filtered_img

def BLPF(image, order, cutoff):
    M, N = image.shape
    H, D = np.zeros((M, N)), np.zeros((M, N))

    U0 = int(M / 2)
    V0 = int(N / 2)

    D0 = cutoff
    n = order

    # For D(u, v)
    for u in range(M):
        for v in range(N):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)

    for u in range(M):
        for v in range(N):
            H[u, v] = 1 / (1 + (D[np.abs(u - U0), np.abs(v - V0)] / D0) ** (2 * n))

    return H

def degrad_func(image, k):
    M, N = image.shape
    H = np.zeros((M, N))

    U0 = int(M / 2)
    V0 = int(N / 2)

    for u in range(M):
        for v in range(N):
            val = (u - U0) ** 2 + (v - V0) ** 2
            val = val ** (5 / 6)
            H[u, v] = np.exp(-k * val)
    return H

def degradation_function(image, k):
    M, N = image.shape
    H = np.zeros((M, N))

    U0 = int(M / 2)
    V0 = int(N / 2)

    for u in range(M):
        for v in range(N):
            val = (u - U0) ** 2 + (v - V0) ** 2
            if val < k:
                H[u, v] = 255
    return H

def W(H, k):
    dh, dw = H.shape
    out = np.zeros((dh, dw))

    for i in range(dh):
        for j in range(dw):
            out[i, j] = (H[i, j] ** 2) / (H[i, j] * ((H[i, j] ** 2) + k))

    return out

def Zero_add_value(H):
    h, w = H.shape
    for i in range(h):
        for j in range(w):
            if H[i, j] == 0:
                H[i, j] += 1e-8

    return H


def filtering(mask, masks):
    for mask in masks:
        print('mask : ', bin(mask))

        filtered_img = image_filter(img, mask, 1, adaptive_threshold_block_size[i],
                                    adaptive_threshold_c[j])

        cv2.imwrite(save_dir + '\\' + file_name + '_filtered.jpg', filtered_img)

        sentence = pytesseract.image_to_string(save_dir + '\\' + file_name + '_filtered.jpg')
        # split
        words = sentence.split()
        # lower case
        words = [word.lower() for word in words]
        # remove punctuations signs
        words = [re.sub(r'[^A-Za-z0-9]+', '', word) for word in words]

        count = 0
        for word in words:
            result_word = Word(word)
            result_word = result_word.spellcheck()
            if len(result_word) == 1:
                count += 1

        if corrects < count:
            result_img = filtered_img
            corrects = count
            correctMask = mask
            adaptive_threshold_block_size_best = adaptive_threshold_block_size[i]
            adaptive_threshold_c_best = adaptive_threshold_c[j]

        print('count : ', count)

    return



if __name__ == "__main__":
    """
    image = cv2.imread('Caffe_blur.jpg', 0)

    print('original')
    plt.imshow(image, cmap='gray'), plt.axis('off')
    plt.show()

    M, N = image.shape

    G = np.fft.fft2(image)
    G = np.fft.fftshift(G)

    print('G')
    FE = np.log(np.abs(G))
    plt.imshow(FE.real, cmap='gray')
    plt.show()

    print('H')
    H = degradation_function(image, k=50)
    H = Zero_add_value(H)
    plt.imshow(H, cmap='gray')
    plt.show()

    W = W(H, k=0.025) # k가 높을수록 LPF가 강해지는 느낌


    """
    # k = 0 -> Inverse
    """


    F_hat = G * W
    f_hat = np.fft.fftshift(F_hat)
    f_hat = np.fft.ifft(f_hat)


    f_hat = f_hat.real.astype(np.uint8)
    print('wiener filtering')
    plt.imshow(f_hat, cmap='gray'), plt.axis('off')
    plt.show()

    print('cutoffs')
    cutoffs = [20, 50, 100, 200, 400, 800, 1600]
    for cut in cutoffs:
        blpf = BLPF(F_hat, 5, cut)
        F_ = F_hat * blpf
        f_ = np.fft.fftshift(F_)
        f_ = np.fft.ifft2(f_)

        f_ = np.clip(f_.real, 0, 255)
        f_ = f_.astype(np.uint8)
        plt.imshow(f_, cmap='gray'), plt.axis('off')
        plt.title(cut)
        plt.show()
    """



    # 가우시안 블러 생성기
    image = cv2.imread('test.png', 0)
    kernel_size = [3, 5, 7, 9, 11]
    sigma = [0.25, 0.3, 0.5, 1, 2, 4]

    count = 0
    save_dir = 'C:\\Users\\poor1\\Desktop'

    for i in range(5):
        for j in range(6):
            gauss = gaussian_blur_func(kernel_size[i], sigma[j])
            return_img = GaussianFilter(image, gauss)
            count += 1
            cv2.imwrite(save_dir + '\\ff\\' + str(i) + ', ' + str(j) + '.jpg', return_img)
            print(count)
