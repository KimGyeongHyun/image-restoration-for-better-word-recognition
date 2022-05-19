import pytesseract
import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt

# define
BINARY_STANDARD = 170   # binary image 만들때 화소값 기준   이미지마다 다른 값 대입이 좋아보임     Eq 잘 되면 조절 필요 없을듯
NONE = 0
EROSION = 1
DILATION = 2
OPENING = 3
CLOSING = 4

# 필터를 수행할지 결정하는 플래그
# 아래 순서로 필터링 진행
FIRST_LINEAR_HQ = False
ADAPTIVE = False
MEDIAN = False
flag_homomorphic = True
HOMOMORPHIC = flag_homomorphic
SECOND_LINEAR_HQ = flag_homomorphic
BINARY = False
MORPHOLOGY = False


# 최종 이미지 필터 /   필터링된 이미지 반환
def image_filter(input_img, method=0, k_size=3):
    return_img = input_img.copy()

    """
    
     HQ
     Adapted filter (가우시안 노이즈 제거)
     Homomorphic filtering (LPF 제거)
     Morphology (표면 다듬기) : 유저가 커널 크기 값을 직접 입력해야 출력됨 /   default 이용
     
     
     가우시안노이즈 일 때와, 가우시안 blur, salt and pepper 구분하여 입력받음
     각각의 상태를 flag로 읽어와서 if 로 사용
     아니면 어차피 함수에 값을 입력해야 하니 사용자가 값을 입력하게끔 함     default 사용
     
     
     blur 상태는 글자가 붙어있으므로 open 사용해야 할 듯
     inverse filter 로 restore 가능하다면 수행
     
     
     모든 전처리 과정을 한꺼번에 넣을 필요는 없음
     쓸만한 전처리 과정만 가져오고, 나머지는 버리는게 더 좋음
     앞서 말했듯 flag를 사용하여 어떤 전처리를 쓸건지 가져와야 할듯
     -> 전역변수 bool flag 사용
     
     morphology true 의 경우 종류와 커널사이즈(3) 을 결정해야 함
     어떻게 넣을지 생각 
     1) 전역변수
     2) 직접
     
     salt and pepper noise 가 완벽히 제거되지 않으면 linear HQ 효과없음
     salt and pepper noise 에만 사용하는 HQ 만들기
     
     
     ################################################
     
     blur 처리
     return_img = cv2.GaussianBlur(img, (7, 7), 2)
     
    """

    # 처음 Histogram Equalization 내장함수 사용시 글자가 두꺼워지며 부정적인 영향을 줌
    # 선형 HQ 사용
    if FIRST_LINEAR_HQ:
        return_img = set_value_to_min_max(return_img)

    # gaussian noise 있을 경우 사용    /   잘못 사용시 글씨가 blur 처리 됨
    # noise_var = 100 일 때도 글자가 배경색이랑 비슷하면 blur 처리 됨
    # noise_var 을 줄여 글자가 blur 처리 안 되게 해야 함
    # gaussian noise를 사용자가 인지하고 입력하게끔 만들 예정
    if ADAPTIVE:
        return_img = adaptive_filtering(return_img, (7, 7), 50)

    # salt and pepper noise 제거      median filter
    # kernel size 5이상이면 글자 손상
    if MEDIAN:
        return_img = median_filter(return_img, 3)

    # Homomorphic filter
    # 효과가 있음, c 값이 높고 high 값이 높으면 다 없어질 때가 있음
    # 전반적으로 검은글씨는 가늘어지고, 흰색글씨는 두꺼워짐
    if HOMOMORPHIC:
        return_img = HF(return_img, cutoff=2, high=1.2, low=0.9, c=20)

    # HQ
    # salt and pepper noise가 완벽히 제거되지 않으면 효과없음  대안 필요
    if SECOND_LINEAR_HQ:
        return_img = set_value_to_min_max(return_img)
    # return_img = cv2.equalizeHist(return_img)     #linear하지 않음

    # Set binary image
    # 128 기준으로 정확도가 떨어지는 문제 발생  /   조정
    if BINARY:
        return_img = get_binary_image(return_img)

    # Morphology
    # binary image 에서만 test 가능
    # image_filter 함수에서 method, k_size 입력이 없을시 수행되지 않음
    # 글자가 작으면 kernel size 3이어도 글자 손상 발생
    # 글자가 적당히 크고 얇고 끊어질 때만 사용 가능할듯  /   binary 단계에서까지 글자가 손상됨
    if MORPHOLOGY:
        return_img = morphology(return_img, method, k_size)

    print("Print image")
    cv2.imshow('image', return_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return return_img


# 테두리 패딩해주는 함수
def Padding(input_img, filter_size):
    print("Padding")
    iw, ih = input_img.shape

    fw, fh = filter_size
    pw, ph = fw // 2, fh // 2  # 소수점 날림

    out = np.zeros((iw + fw - 1, ih + fh - 1))
    out[pw:pw + iw, ph:ph + ih] = input_img

    return out


def adaptive_filtering(input_img, filter_size, noise_var):
    print("Adaptive filter...")
    iw, ih = input_img.shape
    fw, fh = filter_size

    image_padded = Padding(input_img, (fw, fh))
    out = np.zeros((iw, ih))

    for i in range(fw // 2, iw + fw // 2):
        for j in range(fh // 2, ih + fh // 2):
            local = image_padded[i - fw // 2:i + fw // 2 + 1, j - fh // 2:j + fh // 2 + 1]
            local_mean = np.mean(local)
            local_var = np.var(local) + 1e-8  # 0 나누기 대비용 보완 조금만 더함

            if noise_var <= local_var:
                ratio = noise_var / local_var
            else:
                ratio = 1  # 원래 1위로 안 올라감
            out[i - fw // 2, j - fw // 2] = image_padded[i, j] - ratio * (image_padded[i, j] - local_mean)

    out = out.astype(np.uint8)
    return out


def HF(input_img, cutoff, low, high, c):  # Homomorphic filter
    print("Homomorphic filter...")
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
    # gamma h : high, gamma L : low
    # d0 : cutoff
    h, d = np.zeros((p, q)), np.zeros((p, q))

    u0 = int(p / 2)
    v0 = int(q / 2)

    d0 = cutoff

    # For d(u, v)
    for u in range(p):
        for v in range(q):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            d[u, v] = np.sqrt(u2 + v2)
    # 중앙으로부터의 거리를 표현하는 배열

    for u in range(p):
        for v in range(q):
            e = np.exp(-c * ((d[np.abs(u - u0), np.abs(v - v0)] ** 2) / d0 ** 2))
            h[u, v] = (high - low) * (1 - e) + low
    # (1 - e) 부분은 GHPF부분입니다
    # GHPF를 얼마나 적용할지 결정하고 (high-low)
    # 저주파를 얼마나 줄일지 결정합니다 (low)
    # 고주파는 high만큼 증폭됩니다
    # plt.imshow(h, cmap='gray'), plt.axis('off')
    # plt.show()

    g = np.multiply(dft2d, h)
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

    return idft2d


def morphology(input_img, method, k_size=3):
    """
    :param input_img: input_img image
    :param method: 1(erosion), 2(dilation), 3(opening), 4(closing)
    :param k_size: kernel size
    :return:
    """

    h, w = input_img.shape
    pad = k_size // 2  # 몫 연산
    # 커널크기의 1/2 로 제로패딩
    pad_img = np.pad(input_img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    kernel = np.ones((k_size, k_size))
    morp_result = input_img.copy()  # original image init
    # copy() 함수는 원래 이미지에 영향을 주지 않게 하기 위해 만들어짐

    # Erosion or Dilation
    if method == 1 or method == 2:
        for i in range(h):
            for j in range(w):
                if method == 1:
                    morp_result[i, j] = erosion(pad_img[i:i + k_size, j:j + k_size], kernel)
                elif method == 2:
                    morp_result[i, j] = dilation(pad_img[i:i + k_size, j:j + k_size], kernel)
    elif method == 3:  # opening
        morp_result = opening(input_img, k_size)
    elif method == 4:  # closing
        morp_result = closing(input_img, k_size)

    if method == 1:
        print("erosion...")
    elif method == 2:
        print("dilation...")
    elif method == 3:
        print("opening...")
    elif method == 4:
        print("closing...")
    else:
        print("No morphology filter...")

    return morp_result


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


def set_value_to_min_max(input_img):
    print("HQ...")
    return_img = input_img.copy()
    h, w = input_img.shape
    min_value = 0
    max_value = 255
    img_min = np.min(input_img)  # 100
    img_max = np.max(input_img)  # 200

    # 예외처리
    if img_max - img_min == 0:
        return input_img

    for i in range(h):
        for j in range(w):
            return_img[i, j] = (input_img[i, j] - img_min) * ((max_value - min_value) / (img_max - img_min))

    return_img = return_img.astype(np.uint8)

    return return_img


def get_binary_image(input_img):
    print("Get binary image...")
    return_img = input_img.copy()
    h, w = input_img.shape

    for i in range(h):
        for j in range(w):
            if input_img[i, j] < BINARY_STANDARD:
                return_img[i, j] = 0
            else:
                return_img[i, j] = 255

    # cv2.imshow('binary', return_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return return_img.astype(np.uint8)


def median_filter(input_img, kernel_size):
    print("median filter...")
    return_img = input_img.copy()
    m, n = input_img.shape
    pad_size = kernel_size//2
    padded_img = np.zeros((m + 2 * pad_size, n + 2 * pad_size))

    for i in range(m):
        for j in range(n):
            padded_img[i + pad_size, j + pad_size] = input_img[i, j]

    for i in range(m):  # 최대 m-1
        for j in range(n):
            partlist = np.zeros((kernel_size * kernel_size))
            for x in range(kernel_size):    # 최대 kernel_size - 1
                for y in range(kernel_size):
                    partlist[x + y * kernel_size] = padded_img[i + x, j + y]    # 최대 m + kernel_size - 2
            partlist.sort()
            return_img[i, j] = partlist[(kernel_size * kernel_size) // 2]

    return return_img


if __name__ == "__main__":

    # 이미지를 읽어올 주소
    path_dir = 'C:\\Users\\poor1\\Desktop\\scan_folder'
    # 변환된 이미지를 저장하고 텍스트를 추출해서 저장할 주소
    save_dir = 'C:\\Users\\poor1\\Desktop\\filtered_image_save'
    file_list = os.listdir(path_dir)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    result = open(save_dir + '\\output.txt', 'w', encoding='UTF-8')

    for file_name in file_list:
        img = cv2.imread(path_dir + '\\' + file_name, cv2.IMREAD_GRAYSCALE)

        # img 변수 뒤에 다른 변수 없으면 morphology filter 수행 금지
        # 필터링된 이미지를 result_img에 저장하고 cv2로 출력
        result_img = image_filter(img, CLOSING)
        # 이미지 저장
        cv2.imwrite(save_dir + '\\' + file_name + '_filtered.jpg', result_img)

        # 필터링된 이미지에서 텍스트를 추출해서 output.txt에 작성
        result.write(pytesseract.image_to_string(save_dir + '\\' + file_name + '_filtered.jpg', lang='ENG',
                                                 config='--psm 4 -c preserve_interword_spaces=1') + '\n')
    result.close()
    print("complete")
