import cv2
import PIL
from scipy import ndimage
import numpy as np
import os
from math import sqrt, atan, pi, atan2
from sys import stderr


def find_longest_dicks(contours) -> list:  # distance in contours klist
    full_list = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, 0.02 * peri, True)
        for lin_i in range(len(poly))[:-1]:
            full_list.append(((poly[lin_i][0][0], poly[lin_i][0][1]), (poly[lin_i + 1][0][0], poly[lin_i + 1][0][1])))
    full_list = sorted(full_list, key=lambda x: -(x[0][0] - x[1][0]) ** 2 - (x[0][1] - x[1][1]) ** 2)
    return full_list


# k = (y1 - y2) / (x1 - x2)
# b = y2 - k*x2

def line_koeff(line: tuple[tuple[int, int], tuple[int, int]]) -> tuple[float, float]:
    k = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
    b = line[1][1] - k * line[1][0]
    return k, b


def angle_diff(lhs, rhs) -> float:
    k_l, _ = line_koeff(lhs)
    k_r, _ = line_koeff(rhs)
    return abs(atan(k_l) - atan(k_r))


def line_len_f(x):
    return sqrt((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2)


sharpen_cnn = np.array([[0, 0, 0, -1, 0, 0, 0],
                        [0, 0, -1, -1, -1, 0, 0],
                        [0, -1, -1, -1, -1, -1, 0],
                        [-1, -1, -1, 24, -1, -1, -1],
                        [0, -1, -1, -1, -1, -1, 0],
                        [0, 0, -1, -1, -1, 0, 0],
                        [0, 0, 0, -1, 0, 0, 0]])

len_threshold_1 = 0.40
len_threshold_other = 0.8
angle_diff_threshold = 10 * (pi / 180)
OUT_W, OUT_H = 600, 100


# TODO compare results with and without shadow removal
def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


def find_candidate_lines(image_data) -> list:
    # gray = cv2.cvtColor(shadow_remove(image_data), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)  # TODO cmp

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=21)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=21)

    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    grad_zero = grad_norm.copy()
    grad_zero[np.abs(grad_zero) < 64] = 0
    grad_zero = grad_zero.astype(np.float)
    grad_cnn = ndimage.convolve(grad_zero, sharpen_cnn, mode='constant', cval=0.0)
    grad_cnn[grad_cnn > 255] = 255
    grad_cnn[grad_cnn < 64] = 0
    grad_cnn = grad_cnn.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(grad_cnn, cv2.MORPH_CLOSE, kernel)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_dicks = find_longest_dicks(cnts)
    longest = all_dicks[:min(10, len(all_dicks))]

    # here we filter dicks that are not big enough
    if len(longest) <= 2:
        return longest
    target = list()
    target.append(longest[0])  # at least 1 should be here
    current_thr = len_threshold_1
    for line in longest[1:]:
        prev_len = line_len_f(target[-1])
        current_len = line_len_f(line)
        if current_len < prev_len * current_thr:
            break
        if angle_diff(target[0], line) > angle_diff_threshold:
            continue
        target.append(line)
        current_thr = len_threshold_other

    return target


def restrict(x, mn, mx):
    return int(max(mn, min(mx, x)))


def find_intersection(k1: float, b1: float, k2: float, b2: float, x1: float, size: int, recursive: bool = True) -> \
tuple[float, float]:
    y1 = k1 * x1 + b1
    x_r = (y1 + x1 / k1 - b2) / (k2 + 1 / k1)
    y_r = x_r * k2 + b2
    print(k1, b1, k2, b2, x1, x_r, y_r)
    if x_r < -1:
        if not recursive:
            return int(0), int(b2)
        # return find_intersection(k1, b1, k2, b2, x1 - x_r, size)
        res = find_intersection(k2, b2, k1, b1, 1, size, False)
        return find_intersection(k1, b1, k2, b2, res[0], size, False)
    elif y_r < -1:
        if not recursive:
            return int(-b2 / k2), int(0)
        # return find_intersection(k1, b1, k2, b2, x1 - y_r / k1, size)
        res = find_intersection(k2, b2, k1, b1, (-b2) / k2, size, False)
        return find_intersection(k1, b1, k2, b2, res[0], size, False)
    elif x_r > size:
        if not recursive:
            return int(size - 1), int((size - 1) * k2 + b2)
        # return find_intersection(k1, b1, k2, b2, x1 + (size - x_r), size)
        res = find_intersection(k2, b2, k1, b1, (size - 1), size, False)
        return find_intersection(k1, b1, k2, b2, res[0], size, False)
    elif y_r > size:
        if not recursive:
            return int((size - 1) - b2 / k2), int((size - 1))
        # return find_intersection(k1, b1, k2, b2, x1 + (size - y_r) / k1, size)
        res = find_intersection(k2, b2, k1, b1, (((size - 1) - b2) / k2), size, False)
        return find_intersection(k1, b1, k2, b2, res[0], size, False)
    return x_r, y_r


def spec_atan2(y, x):
    res = atan2(y, x)
    print(res, y, x)
    return res


def sort_counterclock(pts: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]) -> tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    avg_x, avg_y = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4, (
            pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4
    result = list(pts)
    print("Points: ", result)
    result = sorted(result, key=lambda pt: atan2((pt[0] - avg_x), (pt[1] - avg_y)))
    # result = sorted(result, key=lambda pt: spec_atan2((pt[1] - avg_y), (pt[0] - avg_x)))
    for pt in result:
        print("pt({}, {}) from avg({}, {}): atan2 = {}".format(pt[0], pt[1], avg_x, avg_y,
                                                               atan2((pt[0] - avg_x), (pt[1] - avg_y))))
    if line_len_f((result[0], result[1])) > line_len_f((result[1], result[2])):
        return result[1], result[2], result[3], result[0]
    return result[0], result[1], result[2], result[3]


def normalize_fit_tetragon(tetragon: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
                           size: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    result = list()
    for x in tetragon:
        result.append((min(size, max(0, x[0])), min(size, max(0, x[1]))))
    return result[0], result[1], result[2], result[3]


def extract_maiha(image: np.ndarray,
                  tetragon: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]) -> np.ndarray:
    # inp_pts = np.array(tetragon, dtype=float)
    # 0231
    stetra = sort_counterclock(tetragon)
    inp_pts = np.array(stetra, dtype=np.float32)
    outp_pts = np.array([[0, 0], [0, OUT_H], [OUT_W, OUT_H], [OUT_W, 0]], dtype=np.float32)
    print(inp_pts)
    print(outp_pts)
    transform_mat = cv2.getPerspectiveTransform(inp_pts, outp_pts)
    return cv2.warpPerspective(image, transform_mat, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR)


def find_rect_by_two_lines(line_ichi: tuple[tuple[int, int], tuple[int, int]], k2: float, b2: float, size: int) -> \
        tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    k1, b1 = line_koeff(line_ichi)
    sanme_point = find_intersection(k1, b1, k2, b2, line_ichi[0][0], size)
    yonme_point = find_intersection(k1, b1, k2, b2, line_ichi[1][0], size)
    sanme_point = restrict(sanme_point[0], 0, size - 1), restrict(sanme_point[1], 0, size - 1)
    yonme_point = restrict(yonme_point[0], 0, size - 1), restrict(yonme_point[1], 0, size - 1)
    return normalize_fit_tetragon((line_ichi[0], line_ichi[1], sanme_point, yonme_point), size)


# there we have two lines
# with that information we can find intersections
# size is the image limit
def find_tetragon_hinted(line_ichi: tuple[tuple[int, int], tuple[int, int]],
                         line_ni: tuple[tuple[int, int], tuple[int, int]], size: int) -> tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    k2, b2 = line_koeff(line_ni)
    return find_rect_by_two_lines(line_ichi, k2, b2, size)


def get_rect_hinted(image: np.ndarray, line_ichi: tuple[tuple[int, int], tuple[int, int]],
                    line_ni: tuple[tuple[int, int], tuple[int, int]], size: int) -> np.ndarray:
    return extract_maiha(image, find_tetragon_hinted(line_ichi, line_ni, size))


def is_valid_rect(rect: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]], size: int) -> bool:
    for x in rect:
        if x[0] >= size or x[1] >= size or x[0] < 0 or x[1] < 0:
            return False
    return True


def rm_background(image: np.ndarray) -> np.ndarray:
    lower = np.array([128, 64, 128])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = 255 - morph
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def compute_image_score(image: np.ndarray, hght: float) -> float:  # we should keep it as low as possible
    # no_green_image: np.ndarray = image.copy()
    # no_green_image[:, :, 1] = 0
    # return float(np.sum(image, dtype=float) + 1) / (image.shape[0] * image.shape[1] * image.shape[2])
    blured = cv2.blur(image, (9, 9))
    nonz = float(np.count_nonzero(blured))
    all_pix = blured.shape[0] * blured.shape[1] * blured.shape[2]
    score = -((hght) ** (1)) + (((all_pix - nonz + 1) / all_pix * 256) ** 2)

    # cv2.imshow('score {}, h:{}, nonz: {}, all: {}'.format(int(score), int(hght), int(nonz), int(all_pix)), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return score  # return black only


# just for testing purposes
def find_rect_no_hint(image: np.ndarray, line: tuple[tuple[int, int], tuple[int, int]], size: int) -> tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    b_step = 2 ** 8
    k1, b1 = line_koeff(line)
    result: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] = (0, 0), (0, 0), (0, 0), (0, 0)
    k_c, b_c = k1, b1
    scoring_image = rm_background(image)
    while b_step >= 1:
        kl, bl = k_c, b_c + b_step  # low
        kh, bh = k_c, b_c - b_step  # high
        rect_l, rect_h = find_rect_by_two_lines(line, kl, bl, size), find_rect_by_two_lines(line, kh, bh, size)
        img_l = extract_maiha(scoring_image, rect_l)
        img_h = extract_maiha(scoring_image, rect_h)
        score_l = compute_image_score(img_l, abs(b1 - bl))  # TODO NEEDS HEAVY ADJUSTMENTS
        score_h = compute_image_score(img_h, abs(b1 - bh))  # CURRENT BUG!!!!!!
        if score_l < score_h:
            k_c, b_c = kl, bl
            result = rect_l
        else:
            k_c, b_c = kh, bh
            result = rect_h
        b_step = int(b_step / 2)
    return result


def get_rect_no_hint(image: np.ndarray, line: tuple[tuple[int, int], tuple[int, int]], size: int) -> np.ndarray:
    return None


def process_image(image: np.ndarray, size: int) -> np.ndarray:
    lines = find_candidate_lines(image)
    if 0 and len(lines) > 1:
        return get_rect_hinted(image, lines[0], lines[1], size)
    elif len(lines) >= 1:
        return get_rect_no_hint(image, lines[0], size)
    return cv2.resize(image, (OUT_W, OUT_H))


def process_image_get_lines(image: np.ndarray, size: int) -> tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    lines = find_candidate_lines(image)
    if 0 and len(lines) > 1:
        res = find_tetragon_hinted(lines[0], lines[1], size)
        for x in res:
            if x[0] >= size or x[1] >= size or x[0] < 0 or x[1] < 0:
                print("hinted failure", x[0], x[1], file=stderr)
        return res
    elif len(lines) >= 1:
        res = find_rect_no_hint(image, lines[0], size)
        for x in res:
            if x[0] >= size or x[1] >= size or x[0] < 0 or x[1] < 0:
                print("no hint failure", x[0], x[1], file=stderr)
        return res
    return (0, 0), (0, size), (size, size), (size, 0)


def show_draw_lines(path: str):
    image = cv2.imread(path)
    image = cv2.resize(image, (1600, 1600))
    lines = find_candidate_lines(image)
    for line in lines:
        cv2.line(image, line[0], line[1], color=(255, 0, 0), thickness=2)

    PIL.Image.fromarray(image).show()


def process_folder_draw_lines(folder: str, target_folder: str):
    files = os.listdir(folder)
    for filename in files[:100]:
        image = cv2.imread(os.path.join(folder, filename))
        image = cv2.resize(image, (1024, 1024))
        lines = find_candidate_lines(image)
        for line in lines:
            cv2.line(image, line[0], line[1], color=(0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(target_folder, filename), image)


def process_folder_draw_tetra(folder: str, target_folder: str):
    files = os.listdir(folder)
    for filename in files[:100]:
        image = cv2.imread(os.path.join(folder, filename))
        image = cv2.resize(image, (1024, 1024))
        print(filename, file=stderr)
        tetra = process_image_get_lines(image, 1024)
        # cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        # print(np.array(tetra))
        cv2.drawContours(image, [np.array(sort_counterclock(tetra), dtype=np.int)], -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(target_folder, filename), image)


def process_folder_extract(folder: str, target_folder: str):
    files = os.listdir(folder)
    for filename in files[:100]:
        image = cv2.imread(os.path.join(folder, filename))
        image = cv2.resize(image, (1024, 1024))
        print(filename, file=stderr)
        extracted = process_image(image, 1024)
        cv2.imwrite(os.path.join(target_folder, filename), extracted)


# show_draw_lines("RiceLeafs/train/Healthy/IMG_20190419_123646.jpg")
# process_folder_draw_lines("RiceLeafs/train/Healthy", "test")
# process_folder_extract("RiceLeafs/train/Healthy", "test")
process_folder_draw_tetra("RiceLeafs/train/Healthy", "test_lines")
