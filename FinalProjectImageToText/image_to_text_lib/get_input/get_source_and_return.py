# 필터 마스크 여러개와 러닝 필터 마스크를 터미널에서 입력받고 변수를 넘겨주는 모듈

# 각 숫자에 따른 필터 dictionary
filter_dic = {
    0: 'adaptive_threshold',
    1: 'first_user_equalization',
    2: 'morphologyEx_opening',
    3: 'median',
    4: 'bilateral',
    5: 'homomorphic',
    6: 'second_user_equalization',
    7: 'gamma_correction',
    8: 'binary',
    9: 'morphology'
}


# 필터 마스크 여러개와 러닝 필터 마스크를 터미널에서 입력받고 변수를 넘겨주는 함수
# 필터 마스크 리스트, 러닝 필터 마스크 리턴
def get_input_source_and_return():

    # # 그라데이션
    # GRAD1 = (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic) | (1 << second_user_equalization)
    # GRAD2 = (1 << homomorphic)
    # GRAD3 = (1 << adaptive_threshold)
    #
    # # 가우시안 노이즈
    # GAUSSIAN1 = (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic) | (
    #         1 << second_user_equalization) | (1 << median)
    # GAUSSIAN2 = (1 << homomorphic)
    #
    # # Salt and pepper 노이즈
    # SAP1 = (1 << median)
    # SAP2 = (1 << first_user_equalization)
    #
    # # blur
    # BLUR1 = (1 << adaptive_threshold)
    # BLUR2 = (1 << homomorphic)
    #
    # # morphology
    # MORP1 = (1 << adaptive_threshold) | (1 << morphology)
    # MORP2 = (1 << binary) | (1 << morphology)
    #
    # # user masks
    # USER_MASK1 = (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic) | (
    #         1 << second_user_equalization) | (1 << median)
    # USER_MASK2 = (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic) | (
    #         1 << second_user_equalization)
    # USER_MASK3 = (1 << adaptive_threshold) | (1 << morphologyEx_opening) | (1 << bilateral)
    # USER_MASK4 = (1 << adaptive_threshold) | (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic)
    # USER_MASK5 = (1 << adaptive_threshold) | (1 << morphologyEx_opening) | (1 << bilateral) | (1 << homomorphic) | (
    #         1 << second_user_equalization)
    # USER_MASK6 = (1 << homomorphic) | (1 << gamma_correction)

    # 사용예시
    # 사용하고자 할 필터를 직접 유저 마스크로 제작하거나, 위에 있는 가이드라인 마스크를 집어넣어서 사용
    # masks = [0, ADAPTIVE_THRESHOLD, FIRST_USER_EQUALIZATION, MORPHOLOGYEX_OPENING, MEDIAN, BILATERAL, HOMOMORPHIC,
    #          SECOND_USER_EQUALIZATION, GAMMA_CORRECTION, BINARY, MORPHOLOGY,
    #          BLUR1, BLUR2, USER_MASK1, USER_MASK2, USER_MASK3, USER_MASK4, USER_MASK5, USER_MASK6]
    # learning_mask = (1 << first_user_equalization_learning) | (1 << first_user_equalization_learning) | (
    #         1 << homomorphic_learning) | (1 << morphology_learning)

    # default
    # masks = [0]
    # learning_mask = 0

    # 구현된 필터
    # 1) 0 : 필터 X
    # 2) GAUSSIAN2 : homomorphic
    # 3) BLUR1 : adaptive_threshold
    # 위의 3개의 필터중 spelling 이 맞는 글자가 많은 필터가 채택되고 구현됨
    # 러닝 필터 : homomorphic, adaptive_threshold
    # masks = [0, BLUR1, GAUSSIAN2]
    # learning_mask = (1 << adaptive_threshold_learning)

    print('-- 이미지에서 글자를 추출하는 프로그램 --')
    print('- 먼저 scan_folter 안에 글자를 추출할 이미지를 넣어주세요 -\n')

    print('사용하고 싶은 필터를 숫자로 하나씩 입력하세요.\n'
          '필터 마스크를 추가하려면 -1 을 입력하세요\n'
          '다 입력했다면 -2 을 입력하세요(-1로 모두 저장 후 입력하세요.)\n\n')

    input_number_list = []  # 필터 넘버링을 임시로 저장
    masks = []              # 필터 마스크들을 저장
    while True:

        print('---필터 종류---\n'
              'adaptive_threshold = 0\n'
              'first_user_equalization = 1\n'
              'morphologyEx_opening = 2\n'
              'median = 3\n'
              'bilateral = 4\n'
              'homomorphic = 5\n'
              'second_user_equalization = 6\n'
              'gamma_correction = 7\n'
              'binary = 8\n'
              'morphology = 9\n')

        # 터미널 창에서 사용할 필터를 숫자로 입력받음
        # 숫자가 아닐 경우 예외처리 후 처음으로 돌아감
        try:
            input_number = int(input('사용할 필터 : '))
        except ValueError:
            print('잘못된 값을 입력했습니다. 사용할 필터를 다시 입력하세요.')
            print('필터 리스트 : ')
            for numbers in input_number_list:
                print('{}'.format(filter_dic[numbers]))
            print('\n')
            continue

        # 입력에 사용되는 수 모음
        if input_number not in [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            continue

        # 필터 마스크 추가
        if input_number == -1:
            mask = 0
            for numbers in input_number_list:   # 비트 마스킹
                mask |= (1 << numbers)
            masks.append(mask)

            # 지금까지 생성된 필터 마스크들 출력
            print('필터 마스크 리스트 : ')
            for mask in masks:
                print('{}'.format(bin(mask)))
            print('새로운 필터 마스크를 만드세요.\n'
                  '다 만들었다면 -2를 입력하세요.\n\n')

            input_number_list = []  # 필터 넘버링 저장 리스트 초기화
            continue

        # 필터 입력 종료
        if input_number == -2:
            break

        # 입력한 수가 중복된 수일 경우
        if input_number in input_number_list:
            print('{}는 중복된 필터입니다.\n'.format(filter_dic[input_number]))
            input_number_list.remove(input_number)  # 해당 수를 지움

        # 입력받은 수를 리스트에 추가 후 정렬
        input_number_list.append(input_number)
        input_number_list.sort()

        # 해당 필터 마스크에서 지금까지 입력받은 필터를 모두 출력
        print('필터 리스트 : ')
        for numbers in input_number_list:
            print('{}'.format(filter_dic[numbers]))
        print('\n')

    print('필터 입력이 끝났습니다.')
    print('필터 마스크 리스트 : ')
    for mask in masks:
        print('{}'.format(bin(mask)))
    print('\n')

    # 러닝 필터 마스크
    learning_input_number_list = []     # 러닝 필터 넘버링을 임시로 저장
    print('러닝을 사용할 필터를 숫자로 하나씩 입력하세요.\n'
          '다 입력했다면 -1 을 입력하세요\n\n')

    while True:
        print('---필터 종류---\n'
              'adaptive_threshold = 0\n'
              'first_user_equalization = 1\n'
              'morphologyEx_opening = 2\n'
              'median = 3\n'
              'bilateral = 4\n'
              'homomorphic = 5\n'
              'second_user_equalization = 6\n'
              'gamma_correction = 7\n'
              'binary = 8\n'
              'morphology = 9\n')

        # 터미널 창에서 사용할 필터를 숫자로 입력받음
        # 숫자가 아닐 경우 예외처리 후 처음으로 돌아감
        try:
            input_number = int(input('사용할 필터 : '))
        except:
            print('필터 리스트 : ')
            for numbers in learning_input_number_list:
                print('{}'.format(filter_dic[numbers]))
            print('\n')
            continue

        # 입력에 사용되는 수 모음
        if input_number not in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            continue

        # 필터 입력 종료
        if input_number == -1:
            break

        # 입력한 수가 중복된 수일 경우
        if input_number in learning_input_number_list:
            print('{}는 중복된 필터입니다.\n'.format(filter_dic[input_number]))
            learning_input_number_list.remove(input_number)     # 해당 수를 지움

        # 입력받은 수를 리스트에 추가 후 정렬
        learning_input_number_list.append(input_number)
        learning_input_number_list.sort()

        # 해당 필터가 러닝을 지원하지 않을 경우
        if input_number in [2, 4]:
            print('해당 필터는 러닝을 지원하지 않습니다.')
            learning_input_number_list.remove(input_number)     # 해당 수를 지움

        # 러닝 필터 마스크 출력
        print('러닝 필터 리스트 : ')
        for numbers in learning_input_number_list:
            print('{}'.format(filter_dic[numbers]))
        print('\n')

    print('필터 입력이 끝났습니다.\n'
          '--------------------------------\n\n')

    # 러닝 필터 비트 마스킹
    learning_mask = 0
    for number in learning_input_number_list:
        learning_mask |= (1 << number)

    # 필터 마스크들과 러닝 필터 마스크 출력
    print('필터 마스크 리스트 : ')
    for mask in masks:
        print('{}'.format(bin(mask)))
    print('\n')
    print('러닝 필터 리스트 : ')
    for numbers in learning_input_number_list:
        print('{}'.format(filter_dic[numbers]))
    print('\n')

    return masks, learning_mask     # 필터 마스크들과 러닝 필터 마스크 리턴


# 테스트 코드
if __name__ == '__main__':
    get_input_source_and_return()
