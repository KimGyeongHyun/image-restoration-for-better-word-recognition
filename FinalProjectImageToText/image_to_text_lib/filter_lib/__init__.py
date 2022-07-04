import pytesseract

# define
NONE = 0
EROSION = 1
DILATION = 2
OPENING = 3
CLOSING = 4

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

# 마스킹
ADAPTIVE_THRESHOLD = 0b00_0000_0001
FIRST_USER_EQUALIZATION = 0b00_0000_0010
MORPHOLOGYEX_OPENING = 0b00_0000_0100
MEDIAN = 0b00_0000_1000
BILATERAL = 0b00_0001_0000
HOMOMORPHIC = 0b00_0010_0000
SECOND_USER_EQUALIZATION = 0b00_0100_0000
GAMMA_CORRECTION = 0b00_1000_0000
BINARY = 0b01_0000_0000
MORPHOLOGY = 0b10_0000_0000

ADAPTIVE_LEARNING = 0b00_0000_0001
USER_SAP_LEARNING = 0b00_0000_0010
MEDIAN_REP_LEARNING = 0b00_0000_1000
HOMO_LEARNING = 0b00_0010_0000
USER_SECOND_LEARNING = 0b00_0100_0000
GAMMA_LEARNING = 0b00_1000_0000
BINARY_LEARNING = 0b01_0000_0000
MORPHOLOGY_LEARNING = 0b10_0000_0000

# 필터별로 러닝할 변수
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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

count = 0  # 어법에 맞는 글자를 세는 변수
corrects = 0  # 각 필터 러닝에서 가장 높은 count 를 저장.
best_counts = 0  # 각 필터 마스크의 카운트 중 가장 높은 count 를 저장.