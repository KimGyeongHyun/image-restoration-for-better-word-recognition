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


def get_input_source_and_return():
    print('-- 이미지에서 글자를 추출하는 프로그램 --')
    print('- 먼저 scan_folter 안에 글자를 추출할 이미지를 넣어주세요 -\n')

    input_number_list = []
    print('사용하고 싶은 필터를 숫자로 하나씩 입력하세요.\n'
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

        try:
            input_number = int(input('사용할 필터 : '))
        except:

            print('필터 리스트 : ')
            for numbers in input_number_list:
                print('{}'.format(filter_dic[numbers]))
            print('\n')
            continue

        if input_number not in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            continue

        # 필터 입력 중단
        if input_number == -1:
            break

        if input_number in input_number_list:
            print('{}는 중복된 필터입니다.\n'.format(filter_dic[input_number]))
            input_number_list.remove(input_number)

        input_number_list.append(input_number)
        input_number_list.sort()

        print('필터 리스트 : ')

        for numbers in input_number_list:
            print('{}'.format(filter_dic[numbers]))
        print('\n')

    print('필터 입력이 끝났습니다.')
    print('필터 리스트 : ')
    for numbers in input_number_list:
        print('{}'.format(filter_dic[numbers]))
    print('\n')

    # 러닝 필터
    learning_input_number_list = []
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

        try:
            input_number = int(input('사용할 필터 : '))
        except:
            print('필터 리스트 : ')
            for numbers in learning_input_number_list:
                print('{}'.format(filter_dic[numbers]))
            print('\n')
            continue

        if input_number not in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            continue

        # 필터 입력 중단
        if input_number == -1:
            break

        if input_number in learning_input_number_list:
            print('{}는 중복된 필터입니다.\n'.format(filter_dic[input_number]))
            learning_input_number_list.remove(input_number)

        learning_input_number_list.append(input_number)
        learning_input_number_list.sort()

        if input_number in [2, 4]:
            print('해당 필터는 러닝을 지원하지 않습니다.')
            learning_input_number_list.remove(input_number)

        print('러닝 필터 리스트 : ')

        for numbers in learning_input_number_list:
            print('{}'.format(filter_dic[numbers]))
        print('\n')

    print('필터 입력이 끝났습니다.\n'
          '--------------------------------\n\n')

    print('필터 리스트 : ')
    for numbers in input_number_list:
        print('{}'.format(filter_dic[numbers]))
    print('\n')
    print('러닝 필터 리스트 : ')
    for numbers in learning_input_number_list:
        print('{}'.format(filter_dic[numbers]))
    print('\n')

    print('헛짓거리 ㅅㄱ')


if __name__ == '__main__':
    get_input_source_and_return()
