import pytesseract
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# 최종 이미지 필터 /   필터링된 이미지 반환
def image_filter(input_img, method=0, k_size=1):
    return_img = input_img.copy()

    """
    
     HQ
     Adapted filter (가우시안 노이즈 제거)
     Homomorphic filtering (LPF 제거)
     Morphology (표면 다듬기) : 유저가 커널 크기 값을 직접 입력해야 출력됨 /   default 이용
     
     
     ################################################
     
     blur 처리
     
     return_img = cv2.GaussianBlur(img, (7, 7), 2)
     
    """

    # 처음 Histogram Equalization 내장함수 사용시 글자가 두꺼워지며 부정적인 영향을 줌
    # return_img = cv2.equalizeHist(img)

    # gaussian noise 있을 경우 사용    /   잘못 사용시 글씨가 blur 처리 됨
    # noise_var = 100 일 때도 글자가 배경색이랑 비슷하면 blur 처리 됨
    # noise_var 을 줄여 글자가 blur 처리 안 되게 해야 함
    # return_img = adaptive_filtering(img, (7, 7), 50)

    # 효과가 있음, c 값이 높고 high 값이 높으면 다 없어질 때가 있음
    # 전반적으로 검은글씨는 가늘어지고, 흰색글씨는 두꺼워짐
    return_img = HF(return_img, cutoff=2, high=1.2, low=0.9, c=20)

    # Morphology
    return_img = morphology(return_img, method, k_size)

    return return_img


# 가운데 패딩해주는 함수
def Padding(input_img, filter_size):
    iw, ih = input_img.shape

    fw, fh = filter_size
    pw, ph = fw//2, fh//2   # 소수점 날림

    out = np.zeros((iw + fw - 1, ih +fh - 1))
    out[pw:pw+iw, ph:ph+ih] = input_img

    return out


def adaptive_filtering(input_img, filter_size, noise_var):
    iw, ih = input_img.shape
    fw, fh = filter_size

    image_padded = Padding(input_img, (fw, fh))
    out = np.zeros((iw, ih))

    for i in range(fw//2, iw+fw//2):
        for j in range(fh//2, ih+fh//2):
            local = image_padded[i-fw//2:i+fw//2+1, j-fh//2:j+fh//2+1]
            local_mean = np.mean(local)
            local_var = np.var(local) + 1e-8    # 0 나누기 대비용 보완 조금만 더함

            if noise_var <= local_var:
                ratio = noise_var / local_var
            else:
                ratio = 1   # 원래 1위로 안 올라감
            out[i-fw//2, j-fw//2] = image_padded[i, j] - ratio * (image_padded[i, j] - local_mean)

    out = out.astype(np.uint8)
    return out


def HF(input_img, cutoff, low, high, c):    # Homomorphic filter

    # plt.imshow(input_img, cmap='gray'), plt.axis('off')
    # plt.show()

    m, n = input_img.shape

    # Zero padding
    p, q = 2 * m, 2 * n
    padded_image = np.zeros((p, q))
    padded_image[:m, :n] = input_img

    # log   /   z = ln(f) = ln(i) + ln(r)
    for x in range(p):
        for y in range(q):
            if padded_image[x, y] == 0:  # ln(0)의 경우는 무한대 이므로 따로 지정합니다
                padded_image[x, y] = 0
            else:
                padded_image[x, y] = np.log(np.abs(padded_image[x, y]))

    # Centering
    padded_image_new = np.zeros((p, q))
    for x in range(p):
        for y in range(q):
            padded_image_new[x, y] = padded_image[x, y] * ((-1) ** (x + y))

    # fourier transform /   Z = Fi + Fr
    dft2d = np.fft.fft2(padded_image_new)
    # plt.imshow(dft2d.real, cmap='gray'), plt.axis('off')
    # plt.show()

    ##############################################
    # gamma H : high, gamma L : low
    # D0 : cutoff
    H, D = np.zeros((p, q)), np.zeros((p, q))

    U0 = int(p / 2)
    V0 = int(q / 2)

    D0 = cutoff

    # For D(u, v)
    for u in range(p):
        for v in range(q):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)
    # 중앙으로부터의 거리를 표현하는 배열

    for u in range(p):
        for v in range(q):
            e = np.exp(-c * ((D[np.abs(u-U0), np.abs(v-V0)]**2)/D0**2))
            H[u, v] = (high-low) * (1 - e) + low
    # (1 - e) 부분은 GHPF부분입니다
    # GHPF를 얼마나 적용할지 결정하고 (high-low)
    # 저주파를 얼마나 줄일지 결정합니다 (low)
    # 고주파는 high만큼 증폭됩니다
    # plt.imshow(H, cmap='gray'), plt.axis('off')
    # plt.show()

    g = np.multiply(dft2d, H)
    # plt.imshow(g.real, cmap='gray'), plt.axis('off')
    # plt.show()

    # IDFT  /   s
    idft2d = np.fft.ifft2(g)

    # De-center
    for x in range(p):
        for y in range(q):
            idft2d[x, y] = idft2d[x, y] * ((-1) ** (x + y))

    # exp   /   g = exp(s) = i0 x r0
    idft2d = np.exp(idft2d)

    # Remove zero padding part
    # print('filtered image')
    idft2d = idft2d[:m, :n].real

    for x in range(m):
        for y in range(n):
            if idft2d[x, y] > 255:
                idft2d[x, y] = 255
            elif idft2d[x, y] < 0:
                idft2d[x, y] = 0

    idft2d = idft2d.astype(np.uint8)
    # plt.imshow(idft2d, cmap='gray'), plt.axis('off')
    # plt.show()

    return idft2d


def morphology(input_img, method, k_size):
    """
    :param input_img: input image
    :param method: 1(erosion), 2(dilation), 3(opening), 4(closing)
    :param k_size: kernel size
    :return:
    """

    h, w = input_img.shape
    pad = k_size // 2   # 몫 연산
    # 커널크기의 1/2 로 제로패딩
    pad_img = np.pad(input_img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    kernel = np.ones((k_size, k_size))
    result = input_img.copy()     # original image init
    # copy() 함수는 원래 이미지에 영향을 주지 않게 하기 위해 만들어짐

    # Erosion or Dilation
    if method == 1 or method == 2:
        for i in range(h):
            for j in range(w):
                if method == 1:
                    result[i, j] = erosion(pad_img[i:i + k_size, j:j + k_size], kernel)
                elif method == 2:
                    result[i, j] = dilation(pad_img[i:i + k_size, j:j + k_size], kernel)
    elif method == 3:   # opening
        result = opening(input_img, k_size)
    elif method == 4:   # closing
        result = closing(input_img, k_size)

    return result


def erosion(boundary=None, kernel=None):
    boundary = boundary * kernel
    if np.min(boundary) == 0:
        return 0
    else:
        return 255


def dilation(boundary=None, kernel=None):
    boundary = boundary * kernel
    if np.max(boundary) != 0:
        return 255
    else:
        return 0


def opening(input_img, k_size):
    erosion_img = morphology(input_img, 1, k_size)
    func_opened_img = morphology(erosion_img, 2, k_size)
    return func_opened_img


def closing(input_img, k_size):
    dilation_img = morphology(input_img, 2, k_size)
    func_closed_img = morphology(dilation_img, 1, k_size)
    return func_closed_img


if __name__ == "__main__":

    # 이미지를 읽어올 주소
    path_dir = 'C:\\Users\\poor1\\Desktop\\scan_folder'
    # 변환된 이미지를 저장하고 텍스트를 추출해서 저장할 주소
    save_dir = 'C:\\Users\\poor1\\Desktop\\filtered_image_save'
    file_list = os.listdir(path_dir)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    result = open(save_dir+'\\output.txt', 'w')

    for file_name in file_list:
        img = cv2.imread(path_dir+'\\'+file_name, cv2.IMREAD_GRAYSCALE)

        result_img = image_filter(img)
        cv2.imwrite(save_dir+'\\'+file_name+'_filtered.jpg', result_img)

        result.write(pytesseract.image_to_string(save_dir+'\\'+file_name+'_filtered.jpg', lang='ENG',
                                                 config='--psm 4 -c preserve_interword_spaces=1')+'\n')
    result.close()
    print("complete")
