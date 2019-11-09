"""
Template matching and extracting ROI with the ORB feature detector
in OpenCV4

Demo driver code in __main__

Written by Tiger Nie for Deloitte DAI
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import distance
import cv2


def show(frame):
    # Helper function to display debug images
    plt.imshow(frame)
    plt.show()


def extract_roi(frame, template):
    """
    Params
    ------
    frame (np.array): target image of interest in BGR or grayscale.
    template (np.array): image of a template in BGR or grayscale.

    Returns
    -------
    warped (np.array): region of interest in the target image.
    """
    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 2:
        gray_template = template
    else:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # ORD feature detection
    orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
    kp, des = orb.detectAndCompute(gray, None)
    kp_template, des_template = orb.detectAndCompute(gray_template, None)

    # DEBUG: Visualise keypoints detected by ORB
    # frame_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0),
    #                              flags=cv2.DrawMatchesFlags_DEFAULT)
    # template_kp = cv2.drawKeypoints(template, kp_template, None,
    #                                 color=(0, 255, 0),
    #                                 flags=cv2.DrawMatchesFlags_DEFAULT)

    # Brute force feature matching
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des)
    # Sort them in the order of their distance.
    goodMatches = sorted(matches, key=lambda x: x.distance)[:50]

    # Map template corners onto the frame image
    src_pts = np.float32(
        [kp_template[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = gray_template.shape
    pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # # Draw the bounding box on the original frame
    # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Debug: visualise matches
    # img3 = np.zeros(frame.shape)
    # img3 = cv2.drawMatches(template, kp_template, frame, kp,
    #                        goodMatches, img3, flags=2)

    # Calculate perspective transform of the template found in the frame
    new_dst = dst.reshape(4, 2)

    (tl, tr, br, bl) = new_dst

    max_height = int(max(distance.euclidean(tr, br),
                         distance.euclidean(tl, bl)))
    max_width = int(max(distance.euclidean(bl, br),
                        distance.euclidean(tl, tr)))

    # construct the destination points which will be used to map the screen
    out_dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Calculate transformation matrix
    trans_mat = cv2.getPerspectiveTransform(new_dst, out_dst)

    # Warp the original frame
    warped = cv2.warpPerspective(frame, trans_mat, (max_width, max_height))

    return warped


if __name__ == "__main__":
    # Load images
    img = cv2.imread('test2.jpg')
    template_front = cv2.imread('front_template.png')
    template_back = cv2.imread('back_template.png')

    # Calculate warps of front and back
    front = extract_roi(img, template_front)
    back = extract_roi(img, template_back)

    # Plot results
    fig = plt.figure()
    gs = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.imshow(img)
    ax1.title.set_text("Original")
    ax2.imshow(front)
    ax2.title.set_text("Front")
    ax3.imshow(back)
    ax3.title.set_text("Back")
    plt.show()
