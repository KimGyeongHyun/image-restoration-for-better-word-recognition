import pytesseract
import numpy as np
import cv2
import os
# import math
# import random
# import matplotlib.pyplot as plt
from textblob import Word
import re

# define
NONE = 0
EROSION = 1
DILATION = 2
OPENING = 3
CLOSING = 4

first_adaptive_threshold = 0
first_user_equalization = 1
inner_opening = 2
median = 3
bilateral = 4
homomorphic = 5
second_user_equalization = 6
gamma_correction = 7
binary = 8
morphology = 9

first_adaptive_threshold_learning = 0
first_user_equalization_learning = 1
median_learning = 3
homomorphic_learning = 5
second_user_equalization_learning = 6
gamma_correction_learning = 7
binary_learning = 8
morphology_learning = 9

MASK10 = 0b10_0000_0000
MASK9 = 0b01_0000_0000
MASK8 = 0b00_1000_0000
MASK7 = 0b00_0100_0000
MASK6 = 0b00_0010_0000
MASK5 = 0b00_0001_0000
MASK4 = 0b00_0000_1000
MASK3 = 0b00_0000_0100
MASK2 = 0b00_0000_0010
MASK1 = 0b00_0000_0001

ADAPTIVE_LEARNING = 0b1  # 1
USER_SAP_LEARNING = 0b10  # 2
MEDIAN_REP_LEARNING = 0b1000  # 4
HOMO_LEARNING = 0b10_0000  # 6
USER_SECOND_LEARNING = 0b100_0000  # 7
GAMMA_LEARNING = 0b1000_0000  # 8
BINARY_LEARNING = 0b1_0000_0000  # 9
MORPHOLOGY_LEARNING = 0b10_0000_0000  # 10

# 변수
adaptive_threshold_block_size = [5, 7, 9, 13, 17, 23, 29]
adaptive_threshold_c = [-1, 2, 3, 4, 5]
adaptive_threshold_block_size_best_first = 0
adaptive_threshold_c_best_first = 0

user_max = [0.8, 0.9, 0.95, 0.97, 0.99]
user_min = [0.2, 0.1, 0.05, 0.03, 0.01]
user_max_best_first = 0
user_min_best_first = 0
user_max_best_second = 0
user_min_best_second = 0

median_repeat_times = [1, 2, 3, 4, 5]
median_repeat_time_best = 0

# img4 에서 cutoff 16이상이어야 인식 잘 됐음
# 나머지는 낮은 cutoff에서 잘 됨
# 연산량이 많아서 오래걸림
# homo_cutoffs = [1, 2, 3, 4, 8, 16, 32, 64]
# homo_c = [5, 10, 20, 30, 50, 70]
homo_cutoffs = [1, 2, 4, 8]
homo_c = [5, 10, 20, 30]
homo_cutoff_best = 0
homo_c_best = 0

gamma_param = [1, 2, 3, 4, 5]
gamma_best_param = 0

binary_standard = [90, 120, 150, 170, 190, 220]
binary_best_standard = 0

morphology_best_method = 0

path_dir = 'C:\\Users\\poor1\\Desktop\\scan_folder'
save_dir = 'C:\\Users\\poor1\\Desktop\\filtered_image_save'
count = 0           # 어법에 맞는 글자를 세는 변수
corrects = 0        # 각 필터 러닝에서 가장 높은 count 를 저장.
best_counts = 0     # 각 필터 마스크의 카운트 중 가장 높은 count 를 저장.


# 최종 이미지 필터 /   필터링된 이미지 반환
def image_filter(input_img, flag_value=0b00_0100_11_01_00_0, input_learning_mask=0):
    global corrects
    global count

    global adaptive_threshold_block_size
    global adaptive_threshold_c
    global adaptive_threshold_block_size_best_first
    global adaptive_threshold_c_best_first
    global adaptive_threshold_block_size_best_second
    global adaptive_threshold_c_best_second

    global user_max
    global user_min
    global user_max_best_first
    global user_min_best_first
    global user_max_best_second
    global user_min_best_second

    global median_repeat_times
    global median_repeat_time_best

    global homo_cutoffs
    global homo_c
    global homo_cutoff_best
    global homo_c_best

    global gamma_param
    global gamma_best_param

    global binary_standard
    global binary_best_standard

    global morphology_best_method

    return_img = input_img.copy()

    # flag_value 를 받아와서 13개의 bool 값을 변환해서 flag 에 대입하기

    if (flag_value & MASK1) == MASK1:
        blurAdaptiveThresholdFlag = True
    else:
        blurAdaptiveThresholdFlag = False

    if (flag_value & MASK2) == MASK2:
        userEqualizationSAPFlag = True
    else:
        userEqualizationSAPFlag = False

    if (flag_value & MASK3) == MASK3:
        innerOpeningFlag = True
    else:
        innerOpeningFlag = False

    if (flag_value & MASK4) == MASK4:
        medianFlag = True
    else:
        medianFlag = False

    if (flag_value & MASK5) == MASK5:
        bilateralFilterFlag = True
    else:
        bilateralFilterFlag = False

    if (flag_value & MASK6) == MASK6:
        homomorphicFlag = True
    else:
        homomorphicFlag = False

    if (flag_value & MASK7) == MASK7:
        userEqualizationSecondFlag = True
    else:
        userEqualizationSecondFlag = False

    if (flag_value & MASK8) == MASK8:
        gammaCorrectionFlag = True
    else:
        gammaCorrectionFlag = False

    if (flag_value & MASK9) == MASK9:
        binaryFlag = True
    else:
        binaryFlag = False

    if (flag_value & MASK10) == MASK10:
        morphologyFlag = True
    else:
        morphologyFlag = False

    """
     ################################################
     
     blur 처리
     return_img = cv2.GaussianBlur(img, (7, 7), 2)
     
     #################################################
     
     노이즈마다 다른 알고리즘 사용
     
     - 가우시안 노이즈
        # inner opening -> bilateral -> homo -> user
        
    #################################################
    
    1) adaptive threshold 
        너무 민감함
        c : -1, 0 에선 검은색 배경   /   0, 1은 격자무늬, 다른 이미지에선 노이즈 발생
        c : 2 부터 제대로 처리 됨 /   c 가 커질수록 글자가 희미해짐 배경이 다이나믹하면  같이 읽어버림
        
        결론 : 블러나 단순한 그라디에이션 배경에서만 사용 필요
        
    2) user equalization
        0.01, 0.99 값이 생각보다 결과가 잘나옴  /   글자 인식용은 아닌듯
        가우시안 노이즈와 EQ에 사용
        
        결론 : 필터링 전 작업. 가우시안이나 eq
        
     
    """

    #####################################################################################################
    # blur 제거

    # 적당한 adaptive threshold를 가해서 바이너리 이미지를 얻어냄
    # 바이너리 속성에 의해 저주파가 뭉개지고 글씨가 선명하게 보임
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

    # inner opening 의 경우 SAP는 제거가 되지만 interpolation 필요함
    # 가우시안 노이즈의 경우 격자무늬가 보이지만 어느정도 커버가 되는 것을 확인
    # HE 만 잘 된다면 선명하게 보일듯

    # median filter : salt and pepper noise 제거
    # 글자가 작으면 kernel size 3이어도 손상나는 것을 확인
    # 큰 글자에서만 적용 가능
    # 사실 큰 글자면 salt and pepper noise 는 있으나 마나일듯
    # 얼마나 반복할 건지

    # if adaptiveFlag:

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
        return_img = inner_opening_filter(return_img)

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
    # SAP 제거용으로는 안 좋음
    # inner opening 이 SAP 잘 제거하는 것을 확인
    # 이후에 나오는 노이즈를 잘 잡음

    # Homomorphic filter
    # 효과가 있음, c 값이 높고 high 값이 높으면 다 없어질 때가 있음
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
                    print(words)

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
    # 선형 HE 는 salt and pepper, gaussian noise 이후 eq 완벽히 되지 않음  대안 필요
    # 이유는 글자보다 노이즈의 min, max값이 크기 때문
    # user eq 로 대체
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
    # 128 기준으로 정확도가 떨어지는 문제 발생  /   170 으로 조정

    # adaptive threshold 테스트 결과 : 성능 좋음
    # 어느정도 블러된 이미지도 처리, 저주파 제거, 글자 안쪽은 하얗게 변함
    # salt and pepper의 경우 글자는 잘 보이지만 나머지 부분에 노이즈 심해짐 (글자 잘 보임) / 인식을 못함
    # 가우시안 노이즈의 경우에만 잘 되지 않음

    # if binaryFlag:
    #     return_img = get_binary_image(return_img)
    # psm 11 보다 3이 압도적으로 좋음
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
    # image_filter 함수에서 method, k_size 입력이 없을시 수행되지 않음
    # 글자가 작으면 kernel size 3이어도 글자 손상 발생
    # 글자가 적당히 크고 얇고 끊어질 때만 사용 가능할듯  /   binary 단계에서까지 글자가 손상됨
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


def adaptive_threshold_filter(input_img, block_size=11, c=2):
    print("Adaptive threshold filtering...")
    return_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
    print('block_size : ', block_size)
    print('c : ', c)

    # cv2.imshow('image', return_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return return_img


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

    # cv2.imshow('image', return_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return return_img


def inner_opening_filter(input_img):
    print("Inner opening filter...")
    kernel = np.ones((3, 3), np.uint8)
    return_img = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernel)

    return return_img


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


def bilateral_filter(input_img):
    print("BilateralFilter...")
    return_img = cv2.bilateralFilter(input_img, -1, 10, 10)

    return return_img


def HF(input_img, cutoff=2, c=30, high=1.2, low=0.9):  # Homomorphic filter
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


def print_all(input_file_name, input_mask, input_learning_mask, input_count):

    used_lerning_mask = input_mask & input_learning_mask

    print("--------------------------------------------")
    print(input_file_name)
    print('total words count : ', input_count)
    print("Print image")

    print('final mask : ', bin(input_mask))
    print("Used filters...")

    if (input_mask & MASK1) == MASK1:
        print("First adaptive threshold filter")

    if (input_mask & MASK2) == MASK2:
        print("First user equalization")

    if (input_mask & MASK3) == MASK3:
        print("Inner opening filter")

    if (input_mask & MASK4) == MASK4:
        print("Median filter")

    if (input_mask & MASK5) == MASK5:
        print("Bilateral filter")

    if (input_mask & MASK6) == MASK6:
        print("Homomorphic filter")

    if (input_mask & MASK7) == MASK7:
        print("Second user equalization")

    if (input_mask & MASK8) == MASK8:
        print("Gamma correction")

    if (input_mask & MASK9) == MASK9:
        print("Binary filter")

    if (input_mask & MASK10) == MASK10:
        print("Morphology filter")

    print('\r\nLearning filters...')

    if (used_lerning_mask & ADAPTIVE_LEARNING) == ADAPTIVE_LEARNING:
        print('First adaptive threshold filter')
        print('Adaptive best block size : ', adaptive_threshold_block_size_best_first)
        print('Adaptive best c : ', adaptive_threshold_c_best_first)

    if (used_lerning_mask & USER_SAP_LEARNING) == USER_SAP_LEARNING:
        print('First user eq')
        print('User best max prob : ', user_max_best_first)
        print('User best min prob : ', user_min_best_first)

    if (used_lerning_mask & MEDIAN_REP_LEARNING) == MEDIAN_REP_LEARNING:
        print('Median best repeat times : ', median_repeat_time_best)

    if (used_lerning_mask & HOMO_LEARNING) == HOMO_LEARNING:
        print('Homomorphic best cutoff : ', homo_cutoff_best)
        print('best c : ', homo_c_best)

    if (used_lerning_mask & USER_SECOND_LEARNING) == USER_SECOND_LEARNING:
        print('Second user eq')
        print('User best max prob : ', user_max_best_second)
        print('User best min prob : ', user_min_best_second)

    if (used_lerning_mask & GAMMA_LEARNING) == GAMMA_LEARNING:
        print('Best gamma parameter : ', gamma_best_param)

    if (used_lerning_mask & BINARY_LEARNING) == BINARY_LEARNING:
        print('Best binary standard : ', binary_best_standard)

    if (used_lerning_mask & MORPHOLOGY_LEARNING) == MORPHOLOGY_LEARNING:
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

    return


def make_noise(std, gray):
    h, w = gray.shape
    img_noise = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            make_noise = np.random.normal()
            set_noise = std * make_noise
            img_noise[i, j] = gray[i, j] + set_noise
            if img_noise[i, j] > 255:
                img_noise[i, j] = 255

    return img_noise.astype(np.uint8)


if __name__ == "__main__":

    """
    
    first_adaptive_threshold = 0
    first_user_equalization = 1
    inner_opening = 2
    median = 3
    bilateral = 4
    homomorphic = 5
    second_user_equalization = 6
    gamma_correction = 7
    binary = 8
    morphology = 9
    
    """

    file_list = os.listdir(path_dir)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    result = open(save_dir + '\\final.txt', 'w', encoding='UTF-8')

    for file_name in file_list:
        print('Start filtering -> ', file_name)
        img = cv2.imread(path_dir + '\\' + file_name, cv2.IMREAD_GRAYSCALE)
        # temp = make_noise(25, img)
        # cv2.imwrite(save_dir + '\\' + file_name + '_filtere.jpg', temp)

        result_img = img.copy()
        # img 변수 뒤에 다른 변수 없으면 morphology_filter filter 수행 금지
        # 필터링된 이미지를 result_img에 저장하고 cv2로 출력

        # inner opening -> bilateral -> homo -> user

        # default
        masks = [0]
        learning_mask = 0

        # 모든 필터를 러닝으로 돌리는 마스크
        # masks = [0, MASK1, MASK2, MASK3, MASK4, MASK5, MASK6, MASK8, MASK10]
        # learning_mask = 0b1111111111

        # gradword 이미지 처리 잘 됨   /   5 -> 14
        # gaussian noise 이미지 처리 어느 정도 됨
        # masks = [(1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (1 << second_user_equalization)]
        # learning_mask = (1 << homomorphic_learning) | (1 << second_user_equalization_learning)

        # gradwardblur 이미지 처리 어느정도 됨    /   0 -> 16 (12, 13정도)
        # masks = [(1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
        #             1 << second_user_equalization)]
        # learning_mask = (1 << homomorphic_learning) | (
        #             1 << second_user_equalization_learning)

        # 2) gaussian 이미지 처리 어느정도 됨
        # masks = [(1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
        #         1 << second_user_equalization) | (1 << median)]
        # learning_mask = (1 << homomorphic_learning) | (1 <<
        #                 second_user_equalization_learning) | (1 << median_learning)

        # 호모만 있어도 완벽하게 된다
        # masks = [(1 << homomorphic)]
        # learning_mask = (1 << homomorphic_learning)

        # binary -> morp
        # 오리지널에 비해 인식률이 저조하지만 정확도가 높고 보기 좋아짐    26 -> 21
        # masks = [(1 << binary) | (1 << morphology)]
        # learning_mask = (1 << binary_learning) | (1 << morphology_learning)

        # masks = [(1 << homomorphic), (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (1 << second_user_equalization) | (
        #     1 << median)]
        # learning_mask = (1 << homomorphic_learning) | (1 << second_user_equalization_learning) | (1 << median_learning)

        # masks = [(1 << binary) | (1 << morphology)]
        # learning_mask = (1 << morphology_learning)

        # masks = [(1 << homomorphic) | (1 << gamma_correction)]
        # learning_mask = (1 << homomorphic_learning) | (1 << gamma_correction_learning)

        # masks = [(1 << first_user_equalization)]
        # learning_mask = (1 << first_user_equalization_learning)

        # 그라데이션
        GRAD1 = (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (1 << second_user_equalization)
        GRAD2 = (1 << homomorphic)
        GRAD3 = (1 << first_adaptive_threshold)

        # 가우시안 노이즈
        GAUSSIAN1 = (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
                1 << second_user_equalization) | (1 << median)
        GAUSSIAN2 = (1 << homomorphic)

        # Salt and pepper 노이즈
        SAP1 = (1 << median)
        SAP2 = (1 << first_user_equalization)

        # blur
        BLUR1 = (1 << first_adaptive_threshold)
        BLUR2 = (1 << homomorphic)

        # morphology
        MORP1 = (1 << first_adaptive_threshold) | (1 << morphology)
        MORP2 = (1 << binary) | (1 << morphology)

        # user masks
        USER_MASK1 = (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
                1 << second_user_equalization) | (1 << median)
        USER_MASK2 = (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
                1 << second_user_equalization)
        USER_MASK3 = (1 << first_adaptive_threshold) | (1 << inner_opening) | (1 << bilateral)
        USER_MASK4 = (1 << first_adaptive_threshold) | (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic)
        USER_MASK5 = (1 << first_adaptive_threshold) | (1 << inner_opening) | (1 << bilateral) | (1 << homomorphic) | (
                1 << second_user_equalization)
        USER_MASK6 = (1 << homomorphic) | (1 << gamma_correction)

        # masks = [BLUR1, BLUR2, USER_MASK1, USER_MASK2, USER_MASK3, USER_MASK4, USER_MASK5, USER_MASK6]
        # learning_mask = (1 << first_adaptive_threshold_learning) | (1 << homomorphic_learning) | (
        #             1 << second_user_equalization_learning)

        masks = [(1 << median) | (1 << homomorphic) | (1 << gamma_correction)]
        learning_mask = (1 << median) | (1 << gamma_correction_learning)

        best_counts = 0
        best_mask = 0

        for mask in masks:  # 마스크(일련의 필터), 여러번 돌아감
            print('mask : ', bin(mask))

            temp_result_img = image_filter(img, mask, learning_mask)
            cv2.imshow('image', temp_result_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imwrite(save_dir + '\\' + file_name + '_filtered.jpg', temp_result_img)

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
            print('mask counts : ', count)

            if best_counts <= count:
                best_counts = count
                best_mask = mask
                result_img = temp_result_img

            print("Mask finish -------------------------------\r\n")

        print_all(file_name, best_mask, learning_mask, best_counts)
        temp = pytesseract.image_to_string(save_dir + '\\' + file_name + '_filtered.jpg')
        print(temp)
        # 이미지 저장
        cv2.imwrite(save_dir + '\\' + file_name + '_filtered.jpg', result_img)

        cv2.imshow('image', result_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # 필터링된 이미지에서 텍스트를 추출해서 output.txt에 작성
        result.write(pytesseract.image_to_string(save_dir + '\\' + file_name + '_filtered.jpg', lang='ENG',
                                                 config='--psm 11 -c preserve_interword_spaces=1') + '\n')

    result.close()
    print("complete")

"""
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output.astype(np.uint8)


def make_noise(std, gray):
    h, w = gray.shape
    img_noise = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            make_noise = np.random.normal()
            set_noise = std * make_noise
            img_noise[i, j] = gray[i, j] + set_noise
            if img_noise[i, j] > 255:
                img_noise[i, j] = 255

    return img_noise.astype(np.uint8)


def anti_gaussian(input_img):
    print("Anti gaussian...")
    h, w = input_img.shape
    return_img = np.zeros((h, w), dtype=np.float64)

    for i in range(30):
        return_img += make_noise(15, input_img)
        print(i)

    return_img = return_img/30

    for i in range(h):
        for j in range(w):
            if return_img[i, j] > 255:
                return_img[i, j] = 255
            elif return_img[i, j] < 0:
                return_img[i, j] = 0

    return return_img.astype(np.uint8)
    
    
    
    
    # 필터를 수행할지 결정하는 플래그
    # 아래 순서로 필터링 진행

    firstLinearHEFlag = False
    blurAdaptiveThresholdFlag = False

    adaptiveFlag = False
    innerOpeningFlag = True
    innerClosingFlag = False
    medianFlag = False

    bilateralFilterFlag = True
    homomorphicFlag = True

    gammaCorrectionFlag = False
    userEqualizationSecondFlag = True

    binaryFlag = False
    binaryFlag = False

    morphologyFlag = False
    
    
    
    
    # Wiener filter 에 사용할 커널 사이즈의 가우시안 함수
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
    
    
    
"""

"""

참조

글자를 txt에 저장하는 함수
https://blog.naver.com/PostView.naver?blogId=ssdyka&logNo=222369731677

이미지 노이즈 만드는데 사용한 함수
https://marisara.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-openCV-10-%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%85%B8
%EC%9D%B4%EC%A6%88Gaussian-Noise

spell checker
https://python-bloggers.com/2022/02/spelling-checker-program-in-python/

"""
