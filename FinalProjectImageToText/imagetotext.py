import pytesseract
import cv2
import os
from textblob import Word
import re

import image_to_text_lib
from image_to_text_lib.filter_lib.filters_and_print import image_filter, print_all

path_dir = '..\\scan_folder'
save_dir = '..\\filtered_image_save'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 필터 넘버링
adaptive_threshold = 0
first_user_equalization = 1
morphologyEx_opening = 2
median = 3
bilateral = 4
homomorphic = 5
second_user_equalization = 6
gamma_correction = 7
binary = 8
morphology = 9

adaptive_threshold_learning = 0
first_user_equalization_learning = 1
median_learning = 3
homomorphic_learning = 5
second_user_equalization_learning = 6
gamma_correction_learning = 7
binary_learning = 8
morphology_learning = 9


if __name__ == "__main__":

    """
    
    adaptive_threshold = 0
    first_user_equalization = 1
    morphologyEx_opening = 2
    median = 3
    bilateral = 4
    homomorphic = 5
    second_user_equalization = 6
    gamma_correction = 7
    binary = 8
    morphology = 9
    
    """

    masks, learning_mask = image_to_text_lib.get_input_source_and_return()

    file_list = os.listdir(path_dir)
    result = open(save_dir + '\\final.txt', 'w', encoding='UTF-8')

    for file_name in file_list:
        print('Start filtering -> ', file_name)
        img = cv2.imread(path_dir + '\\' + file_name, cv2.IMREAD_GRAYSCALE)

        result_img = img.copy()

        best_counts = 0
        best_mask = 0

        for mask in masks:  # 마스크(일련의 필터), 여러번 돌아감
            print('mask : ', bin(mask))

            temp_result_img = image_filter(file_name, img, mask, learning_mask)
            # cv2.imshow('image', temp_result_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
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

참조

글자를 txt에 저장하는 함수
https://blog.naver.com/PostView.naver?blogId=ssdyka&logNo=222369731677

이미지 노이즈 만드는데 사용한 함수
https://marisara.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-openCV-10-%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%85%B8
%EC%9D%B4%EC%A6%88Gaussian-Noise

spell checker
https://python-bloggers.com/2022/02/spelling-checker-program-in-python/

"""
