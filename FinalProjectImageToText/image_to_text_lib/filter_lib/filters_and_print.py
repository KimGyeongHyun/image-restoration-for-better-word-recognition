import numpy as np
import cv2
from textblob import Word
import re

from ..filter_lib import *

adaptive_threshold_block_size_best_first = 0
adaptive_threshold_c_best_first = 0

user_max_best_first = 0
user_min_best_first = 0
user_max_best_second = 0
user_min_best_second = 0

median_repeat_time_best = 0

homo_cutoff_best = 0
homo_c_best = 0

gamma_best_param = 0

binary_best_standard = 0

morphology_best_method = 0


# 최종 이미지 필터 /   필터링된 이미지 반환
def image_filter(file_name, input_img, flag_value=0, input_learning_mask=0):
    # flag_value : 필터 종류를 마스킹한 변수
    # input_learning_mask : 러닝할 필터 종류를 마스킹한 변수

    return_img = input_img.copy()
    temp_return_img = input_img.copy()

    global adaptive_threshold_block_size_best_first
    global adaptive_threshold_c_best_first

    global user_max_best_first
    global user_min_best_first
    global user_max_best_second
    global user_min_best_second

    global median_repeat_time_best

    global homo_cutoff_best
    global homo_c_best

    global gamma_best_param

    global binary_best_standard

    global morphology_best_method

    #######################################################################################################
    # 사용하는 필터 정보를 flag_value로 받아와서 필터 사용 여부 결정

    if (flag_value & ADAPTIVE_THRESHOLD) == ADAPTIVE_THRESHOLD:
        blurAdaptiveThresholdFlag = True
    else:
        blurAdaptiveThresholdFlag = False

    if (flag_value & FIRST_USER_EQUALIZATION) == FIRST_USER_EQUALIZATION:
        userEqualizationSAPFlag = True
    else:
        userEqualizationSAPFlag = False

    if (flag_value & MORPHOLOGYEX_OPENING) == MORPHOLOGYEX_OPENING:
        innerOpeningFlag = True
    else:
        innerOpeningFlag = False

    if (flag_value & MEDIAN) == MEDIAN:
        medianFlag = True
    else:
        medianFlag = False

    if (flag_value & BILATERAL) == BILATERAL:
        bilateralFilterFlag = True
    else:
        bilateralFilterFlag = False

    if (flag_value & HOMOMORPHIC) == HOMOMORPHIC:
        homomorphicFlag = True
    else:
        homomorphicFlag = False

    if (flag_value & SECOND_USER_EQUALIZATION) == SECOND_USER_EQUALIZATION:
        userEqualizationSecondFlag = True
    else:
        userEqualizationSecondFlag = False

    if (flag_value & GAMMA_CORRECTION) == GAMMA_CORRECTION:
        gammaCorrectionFlag = True
    else:
        gammaCorrectionFlag = False

    if (flag_value & BINARY) == BINARY:
        binaryFlag = True
    else:
        binaryFlag = False

    if (flag_value & MORPHOLOGY) == MORPHOLOGY:
        morphologyFlag = True
    else:
        morphologyFlag = False

    ####################################################################################################
    # flag 값에 따라 필터 사용 여부 결정
    # 필터가 사용될 예정이라면 그 필터를 러닝할건지 input_learning_mask 로 결정

    #####################################################################################################
    # blur 제거

    # 적당한 adaptive threshold를 가해서 바이너리 이미지를 얻어냄
    # 바이너리 속성에 의해 저주파가 없어지고 글씨가 선명하게 보임
    # 어느정도 블러된 이미지 처리, 저주파 제거
    # 노이즈에 민감, 노이즈가 강조됨 /   노이즈 있는 영상은 사용 불가

    if blurAdaptiveThresholdFlag:
        if (input_learning_mask & ADAPTIVE_LEARNING) == ADAPTIVE_LEARNING:
            corrects = 0
            for i in range(len(adaptive_threshold_block_size)):
                for j in range(len(adaptive_threshold_c)):
                    #########

                    filtered_img = adaptive_threshold_filter(return_img, adaptive_threshold_block_size[i],
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

                    if corrects <= count:
                        temp_return_img = filtered_img
                        corrects = count
                        adaptive_threshold_block_size_best_first = adaptive_threshold_block_size[i]
                        adaptive_threshold_c_best_first = adaptive_threshold_c[j]

                    print('count : ', count, '\r\n')
            print('First adaptive threshold corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = adaptive_threshold_filter(return_img)

    #####################################################################################################
    # Salt and pepper noise 제거

    # inner opening 의 경우 SAP는 제거가 되지만 조금의 노이즈 남음
    # 가우시안 노이즈의 경우 격자무늬가 보이지만 어느정도 커버가 되는 것을 확인
    # bilateral filter 와 같이 사용

    # median filter
    # salt and pepper noise 제거
    # 얼마나 반복할 건지 러닝

    if userEqualizationSAPFlag:
        if (input_learning_mask & USER_SAP_LEARNING) == USER_SAP_LEARNING:
            corrects = 0
            for i in range(len(user_max)):
                for j in range(len(user_min)):

                    filtered_img = user_equalization(return_img, user_min[j], user_max[i])

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

                    if corrects <= count:
                        temp_return_img = filtered_img
                        corrects = count
                        user_max_best_first = user_max[i]
                        user_min_best_first = user_min[j]

                    print('count : ', count, '\r\n')
            print('First user equalization corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = user_equalization(return_img, 0.2, 0.8)

    if innerOpeningFlag:
        return_img = morphologyEx_opening_filter(return_img)

    if medianFlag:
        if (input_learning_mask & MEDIAN_REP_LEARNING) == MEDIAN_REP_LEARNING:
            corrects = 0
            for i in range(len(median_repeat_times)):

                filtered_img = return_img.copy()
                for j in range(median_repeat_times[i]):
                    filtered_img = median_filter(filtered_img)

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

                if corrects <= count:
                    temp_return_img = filtered_img
                    corrects = count
                    median_repeat_time_best = i

                print('count : ', count, '\r\n')
            print('Median filter corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            for i in range(3):
                return_img = median_filter(return_img)

    #####################################################################################################
    # 이후 노이즈 제거

    # 내장함수 bilateral filter 사용
    # inner opening 이후에 나오는 노이즈를 잘 잡음

    # Homomorphic filter
    # 저주파 배경 제거
    # c 값이 높고 high 값이 높으면 다 없어질 때가 있음
    # 전반적으로 검은글씨는 가늘어지고, 흰색글씨는 두꺼워짐

    if bilateralFilterFlag:
        return_img = bilateral_filter(return_img)

    if homomorphicFlag:
        if (input_learning_mask & HOMO_LEARNING) == HOMO_LEARNING:
            corrects = 0
            for i in range(len(homo_cutoffs)):
                for j in range(len(homo_c)):

                    filtered_img = HF(return_img, homo_cutoffs[i], homo_c[j])

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

                    if corrects <= count:
                        temp_return_img = filtered_img
                        corrects = count
                        homo_cutoff_best = homo_cutoffs[i]
                        homo_c_best = homo_c[j]

                    print('count : ', count, '\r\n')
            print('Homomorphic filter corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = HF(return_img)

    #####################################################################################################
    # 두번째 Eq

    # HQ
    # 선형 HE 는 salt and pepper, gaussian noise 이후 eq 완벽히 되지 않음
    # 이유는 글자보다 노이즈의 min, max값이 크기 때문
    # user eq 를 직접 만들어서 선형 HE 대체
    if userEqualizationSecondFlag:
        if (input_learning_mask & USER_SECOND_LEARNING) == USER_SECOND_LEARNING:
            corrects = 0
            for i in range(len(user_max)):
                for j in range(len(user_min)):

                    filtered_img = user_equalization(return_img, user_min[j], user_max[i])

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

                    if corrects <= count:
                        temp_return_img = filtered_img
                        corrects = count
                        user_max_best_second = user_max[i]
                        user_min_best_second = user_min[j]

                    print('count : ', count, '\r\n')
            print('Second user equalization corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = user_equalization(return_img)

    # 검은색 글자를 뚜렷하게 해줌

    if gammaCorrectionFlag:
        if (input_learning_mask & GAMMA_LEARNING) == GAMMA_LEARNING:
            corrects = 0
            for i in range(len(gamma_param)):
                filtered_img = gamma_correction_filter(return_img, gamma_param[i])

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

                if corrects <= count:
                    temp_return_img = filtered_img
                    corrects = count
                    gamma_best_param = gamma_param[i]

                print('count : ', count, '\r\n')
            print('Gamma correction corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = gamma_correction_filter(return_img)

    #####################################################################################################
    # binary 이미지 가져오기

    # Set binary image
    # 128 기준으로 정확도가 떨어지는 문제 발생  /   러닝 사용

    if binaryFlag:
        if (input_learning_mask & BINARY_LEARNING) == BINARY_LEARNING:
            corrects = 0
            for i in range(len(binary_standard)):
                print('binary_standard : ', binary_standard[i])
                filtered_img = binary_filter(return_img, binary_standard[i])

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

                if corrects <= count:
                    temp_return_img = filtered_img
                    corrects = count
                    binary_best_standard = binary_standard[i]

                print('count : ', count, '\r\n')
            return_img = temp_return_img
            print('Binary corrects', corrects, '\r\n')
        else:
            return_img = binary_filter(return_img)

    #####################################################################################################
    # Morphology

    # binary image 에서만 test 가능
    # image_filter 함수에서 method, k_size 입력이 없을시 dilation
    # 글자가 작으면 kernel size 3이어도 글자 손상 발생
    # 글자가 적당히 크고 얇고 끊어질 때만 사용 가능할듯
    # 작은 이미지의 경우 dilation 이 이득으로 나옴

    if morphologyFlag:
        if (input_learning_mask & MORPHOLOGY_LEARNING) == MORPHOLOGY_LEARNING:
            corrects = 0
            for i in range(6):

                filtered_img = morphology_learning_function(return_img, i)

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

                if corrects <= count:
                    temp_return_img = filtered_img
                    corrects = count
                    morphology_best_method = i

                print('count : ', count, '\r\n')
            print('Morphology filter corrects', corrects, '\r\n')
            return_img = temp_return_img
        else:
            return_img = morphology_filter(return_img)

    return return_img


# adaptive thresholding /   default block size = 11, c parameter = 2
def adaptive_threshold_filter(input_img, block_size=11, c=2):
    print("Adaptive threshold filtering...")
    return_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
    print('block_size : ', block_size)
    print('c : ', c)

    return return_img


# linear equalization   /   분포도에 따라 cutoff  /   default 아래쪽 : 3%, 위쪽 : 90%
def user_equalization(input_img, min_thresh_prob=0.03, max_thresh_prob=0.9):
    print("User equalization...")
    h, w = input_img.shape
    return_img = np.zeros((h, w))

    pixel_count = np.zeros(256)
    pixel_number = h * w
    pixel_sum = 0
    user_min_pixel_val = 0
    user_max_pixel_val = 0

    for i in range(h):
        for j in range(w):
            pixel_value = input_img[i, j]
            pixel_count[pixel_value] += 1

    for i in range(256):
        pixel_sum += pixel_count[i]
        if pixel_sum > (pixel_number * min_thresh_prob):
            user_min_pixel_val = i
            break

    pixel_sum = 0
    for i in range(256):
        pixel_sum += pixel_count[i]
        if pixel_sum > (pixel_number * max_thresh_prob):
            user_max_pixel_val = i
            break

    print('user max pixel value: ', user_max_pixel_val)
    print('user min pixel value: ', user_min_pixel_val)
    print('user max prob : ', max_thresh_prob)
    print('user min prob : ', min_thresh_prob)

    if user_max_pixel_val == user_min_pixel_val:
        return np.zeros((h, w))

    min_value = 0
    max_value = 255

    for i in range(h):
        for j in range(w):
            temp_min = input_img[i, j] - user_min_pixel_val
            if temp_min < 0:
                temp_min = 0
            return_img[i, j] = temp_min * ((max_value - min_value) / (user_max_pixel_val - user_min_pixel_val))
            if return_img[i, j] > 255:
                return_img[i, j] = 255

    return_img = return_img.astype(np.uint8)

    return return_img


# morphologyEx opening filter
def morphologyEx_opening_filter(input_img):
    print("Inner opening filter...")
    kernel = np.ones((3, 3), np.uint8)
    return_img = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernel)

    return return_img


# median filter
def median_filter(input_img, kernel_size=3):
    print("median filter...")
    return_img = input_img.copy()
    m, n = input_img.shape
    pad_size = kernel_size // 2
    padded_img = np.zeros((m + 2 * pad_size, n + 2 * pad_size))

    for i in range(m):
        for j in range(n):
            padded_img[i + pad_size, j + pad_size] = input_img[i, j]

    for i in range(m):  # 최대 m-1
        for j in range(n):
            partlist = np.zeros((kernel_size * kernel_size))
            for x in range(kernel_size):  # 최대 kernel_size - 1
                for y in range(kernel_size):
                    partlist[x + y * kernel_size] = padded_img[i + x, j + y]  # 최대 m + kernel_size - 2
            partlist.sort()
            return_img[i, j] = partlist[(kernel_size * kernel_size) // 2]

    return return_img


# bilateral filter
def bilateral_filter(input_img):
    print("BilateralFilter...")
    return_img = cv2.bilateralFilter(input_img, -1, 10, 10)

    return return_img


# homomorphic filter    /   default value cutoff = 2, c parameter = 30
def HF(input_img, cutoff=1, c=30, high=1.2, low=0.9):  # Homomorphic filter
    print("Homomorphic filter...")
    print('cutoff : ', cutoff)
    print('c : ', c)

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


# gamma correction  /   default c parameter = 3
def gamma_correction_filter(input_img, c_param=3):
    print("Gamma correction...")
    print('gamma parameter : ', c_param)
    normalized_img = input_img / 255
    h, w = input_img.shape
    return_img = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            return_img[i, j] = normalized_img[i, j] ** c_param

    return_img = (return_img * 255).astype(np.uint8)

    return return_img


# get binary image  /   default binary standard = 170
def binary_filter(input_img, input_binary_standard=170):
    print("get binary image")
    return_img = input_img.copy()
    h, w = input_img.shape

    for i in range(h):
        for j in range(w):
            if input_img[i, j] < input_binary_standard:
                return_img[i, j] = 0
            else:
                return_img[i, j] = 255

    return return_img.astype(np.uint8)


# morphology filter /   default : dilation
def morphology_filter(input_img, method=2, k_size=3):
    """
    :param input_img: input_img image
    :param method: 1(erosion), 2(dilation), 3(opening), 4(closing)
    :param k_size: kernel size
    :return:
    """

    def erosion(boundary=None, input_kernel=None):
        boundary = boundary * input_kernel
        if np.min(boundary) == 0:
            return 0
        else:
            return 255

    def dilation(boundary=None, input_kernel=None):
        boundary = boundary * input_kernel
        if np.max(boundary) != 0:
            return 255
        else:
            return 0

    def opening(input_input_img, input_k_size):
        erosion_img = morphology_filter(input_input_img, 1, input_k_size)
        func_opened_img = morphology_filter(erosion_img, 2, input_k_size)
        return func_opened_img

    def closing(input_input_img, input_k_size):
        dilation_img = morphology_filter(input_input_img, 2, input_k_size)
        func_closed_img = morphology_filter(dilation_img, 1, input_k_size)
        return func_closed_img

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
        print("No morphology_filter filter...")

    return morp_result


# morphology 반복 학습을 위한 함수
def morphology_learning_function(input_img, case):
    if case == 0:
        return morphology_filter(input_img, EROSION)
    elif case == 1:
        return morphology_filter(input_img, DILATION)
    elif case == 2:
        return morphology_filter(input_img, OPENING)
    elif case == 3:
        return morphology_filter(input_img, CLOSING)
    elif case == 4:
        temp = morphology_filter(input_img, OPENING)
        return morphology_filter(temp, CLOSING)
    elif case == 5:
        temp = morphology_filter(input_img, CLOSING)
        return morphology_filter(temp, OPENING)
    else:
        return input_img


# 모든 정보를 출력하는 함수
def print_all(input_file_name, input_mask, input_learning_mask, input_count):
    used_learning_mask = input_mask & input_learning_mask

    print("--------------------------------------------")
    print(input_file_name)
    print('total words count : ', input_count)
    print("Print image")

    print('final mask : ', bin(input_mask))
    print("Used filters...")

    if (input_mask & ADAPTIVE_THRESHOLD) == ADAPTIVE_THRESHOLD:
        print("First adaptive threshold filter")

    if (input_mask & FIRST_USER_EQUALIZATION) == FIRST_USER_EQUALIZATION:
        print("First user equalization")

    if (input_mask & MORPHOLOGYEX_OPENING) == MORPHOLOGYEX_OPENING:
        print("Inner opening filter")

    if (input_mask & MEDIAN) == MEDIAN:
        print("Median filter")

    if (input_mask & BILATERAL) == BILATERAL:
        print("Bilateral filter")

    if (input_mask & HOMOMORPHIC) == HOMOMORPHIC:
        print("Homomorphic filter")

    if (input_mask & SECOND_USER_EQUALIZATION) == SECOND_USER_EQUALIZATION:
        print("Second user equalization")

    if (input_mask & GAMMA_CORRECTION) == GAMMA_CORRECTION:
        print("Gamma correction")

    if (input_mask & BINARY) == BINARY:
        print("Binary filter")

    if (input_mask & MORPHOLOGY) == MORPHOLOGY:
        print("Morphology filter")

    print('\r\nLearning filters...')

    if (used_learning_mask & ADAPTIVE_LEARNING) == ADAPTIVE_LEARNING:
        print('First adaptive threshold filter')
        print('Adaptive best block size : ', adaptive_threshold_block_size_best_first)
        print('Adaptive best c : ', adaptive_threshold_c_best_first)

    if (used_learning_mask & USER_SAP_LEARNING) == USER_SAP_LEARNING:
        print('First user eq')
        print('User best max prob : ', user_max_best_first)
        print('User best min prob : ', user_min_best_first)

    if (used_learning_mask & MEDIAN_REP_LEARNING) == MEDIAN_REP_LEARNING:
        print('Median best repeat times : ', median_repeat_time_best)

    if (used_learning_mask & HOMO_LEARNING) == HOMO_LEARNING:
        print('Homomorphic best cutoff : ', homo_cutoff_best)
        print('best c : ', homo_c_best)

    if (used_learning_mask & USER_SECOND_LEARNING) == USER_SECOND_LEARNING:
        print('Second user eq')
        print('User best max prob : ', user_max_best_second)
        print('User best min prob : ', user_min_best_second)

    if (used_learning_mask & GAMMA_LEARNING) == GAMMA_LEARNING:
        print('Best gamma parameter : ', gamma_best_param)

    if (used_learning_mask & BINARY_LEARNING) == BINARY_LEARNING:
        print('Best binary standard : ', binary_best_standard)

    if (used_learning_mask & MORPHOLOGY_LEARNING) == MORPHOLOGY_LEARNING:
        if morphology_best_method == 0:
            print('Erosion')
        elif morphology_best_method == 1:
            print('Dilation')
        elif morphology_best_method == 2:
            print('Opening')
        elif morphology_best_method == 3:
            print('Closing')
        elif morphology_best_method == 4:
            print('Opening and closing')
        elif morphology_best_method == 5:
            print('Closing and opening')

    print("--------------------------------------------")
