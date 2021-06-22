import cv2
import numpy as np


def sift_match(img1, img2):

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb_detector = cv2.ORB_create()

    kpt1, des1 = orb_detector.detectAndCompute(img1, None)
    kpt2, des2 = orb_detector.detectAndCompute(img2, None)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=1)

    scores = [m[0].distance for m in matches]
    score_threshold = np.median(scores)

    matches = [m[0] for m in matches if m[0].distance < score_threshold]

    return matches, kpt1, kpt2


def fundamental_matrix_estimate(kpt1, kpt2, matches):
    pts1 = [kpt1[m.queryIdx].pt for m in matches]
    pts2 = [kpt2[m.trainIdx].pt for m in matches]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)

    # filtered_matches = [m for i, m in enumerate(matches) if mask[i][0] == 1]
    # img = cv2.drawMatches(img1data, kpt1, img2data, kpt2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('he', img)
    # cv2.waitKey(0)

    return F, mask


def draw_lines(img, lines, colors=None):
    if colors is None:
        colors = np.random.randint(0, 255, (len(lines), 3))
    x0, x1 = 0, img.shape[1]
    for i, parameter in enumerate(lines):
        a, b, c = parameter
        # a*x + b*y + c = 0
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        cv2.line(img, pt1=(x0, y0), pt2=(x1, y1), color=tuple(colors[i].tolist()), thickness=2)

    return img


def compute_epilines(pts, pts_source, F):
    """ given fundamental matrix F and corresponding points, compute
        epipolar lines in img1 and img2

    Args:
        pts: given points [[x, y],...]
        pts_source: identify pts are from img1 or img2
        F: fundamental matrix

    Returns:
        epipolar lines in img1, and eipolar lines in img2
    """
    # points padding
    pts = np.array(pts)
    pts = np.concatenate((pts, np.ones((len(pts), 1))), axis=1)
    if pts_source == 1:
        lines2 = np.dot(F, pts.T).T.tolist()
        # compute e, F \dot e = 0
        eigvals, eigvecs = np.linalg.eig(F)
        e = eigvecs.T[np.argmin(eigvals)]
        lines1 = [np.cross(pt, e).tolist() for pt in pts]
    else:
        lines1 = np.dot(F.T, pts.T).T.tolist()
        # compute e, F.T \dot e = 0
        eigvals, eigvecs = np.linalg.eig(F.T)
        e = eigvecs.T[np.argmin(eigvals)]
        lines2 = [np.cross(pt, e).tolist() for pt in pts]
    return lines1, lines2


def projection_distance(point, line):
    # point = [x, y], line = [a, b, c]
    x, y = point
    a, b, c = line
    absv = np.abs(a * x + b * y + c)/np.sqrt(a**2 + b**2)
    return absv


def fundamental_matrix_filter(kpt1, kpt2, F, threshold):
    """ given matched points, compute if there are matched according to F matrix

    Args:
        kpt1: keypoints in image1 matched with keypoints in image2
        kpt2: keypoints in image2 matched with keypoints in image1
        F: fundamental matrix

    Returns:
        A bool vector, [True, False, ...]
    """

    pts1 = np.int32(kpt1)
    pts2 = np.int32(kpt2)

    # lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    # lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)

    lines1, lines2 = compute_epilines(pts2, 2, F)
    distance1 = [projection_distance(p, line) for p, line in zip(pts1, lines1)]
    lines1, lines2 = compute_epilines(pts1, 1, F)
    distance2 = [projection_distance(p, line) for p, line in zip(pts2, lines2)]

    distance = np.array([(a + b)/2 for a, b in zip(distance1, distance2)])

    return distance < threshold


if __name__ == '__main__':
    img_name1 = 'data/pair1/img1.png'
    img_name2 = 'data/pair1/img2.png'

    img1 = cv2.imread(img_name1)
    img2 = cv2.imread(img_name2)

    matches, kpt1, kpt2 = sift_match(img1, img2)
    F, mask = fundamental_matrix_estimate(kpt1, kpt2, matches)

    # fundamental_matrix_filter(kpt1, kpt2, F, 2.)

    # random epipolar lines
    pts = np.random.random((10, 2))
    pts[:, 0] = pts[:, 0] * img1.shape[1]
    pts[:, 1] = pts[:, 1] * img1.shape[0]

    lines1, lines2 = compute_epilines(pts, 1, F)
    colors = np.random.randint(0, 255, (len(lines1), 3))
    img1_ = draw_lines(img1.copy(), lines1, colors)
    img2_ = draw_lines(img2.copy(), lines2, colors)

    cv2.imshow('img1', img1_)
    cv2.imshow('img2', img2_)
    cv2.waitKey(0)

    img_match = cv2.drawMatches(img1.copy(), kpt1, img2.copy(), kpt2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('img_match', img_match)
    cv2.waitKey(0)

    kpt1 = np.array([kpt1[m.queryIdx].pt for m in matches])
    kpt2 = np.array([kpt2[m.trainIdx].pt for m in matches])

    mask = fundamental_matrix_filter(kpt1, kpt2, F, 2)

    kpt1 = kpt1[mask]
    kpt2 = kpt2[mask]

    kpt1 = [cv2.KeyPoint(p[0], p[1], 2) for p in kpt1]
    kpt2 = [cv2.KeyPoint(p[0], p[1], 2) for p in kpt2]

    matches = [cv2.DMatch(i, i, 0, 0) for i in range(len(kpt1))]

    img_match_after = cv2.drawMatches(img1, kpt1, img2, kpt2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('img_match_after', img_match_after)
    cv2.waitKey(0)




