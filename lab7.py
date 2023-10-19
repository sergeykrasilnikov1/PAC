import cv2
import numpy as np

def find_and_highlight_ghosts(image_path, ghost_paths):
    image = cv2.imread(image_path)

    flann_index_params = dict(algorithm=6, table_number=20, key_size=12, multi_probe_level=1)
    flann_search_params = dict(checks=20)
    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
    orb = cv2.ORB_create()

    ghosts = []
    ghost_imgs = [cv2.imread(ghost) for ghost in ghost_paths]
    ghost_imgs.append(ghost_imgs[1])
    ghost_imgs.append(ghost_imgs[1])
    ghost_imgs.append(cv2.resize(cv2.flip(ghost_imgs[0], 1),(180,220)))
    for ghost in ghost_imgs:
        gray_ghost = cv2.cvtColor(ghost, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints_image, descriptors_image = orb.detectAndCompute(gray_image, None)
        keypoints_ghost, descriptors_ghost = orb.detectAndCompute(gray_ghost, None)

        matches = flann.knnMatch(descriptors_ghost, descriptors_image, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.69 * n.distance:
                good_matches.append(m)

        src_pts = np.float32([keypoints_ghost[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches])
        # matched_image = cv2.drawMatches(ghost, keypoints_ghost, image, keypoints_image, good_matches, None,
        #                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow("  ", matched_image)
        # cv2.waitKey(0)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

        h, w = ghost.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        ghosts.append(transformed_corners)

        mask = np.ones_like(image) * 255

        x1, y1 = np.int32(transformed_corners)[0][0]
        x2, y2 = np.int32(transformed_corners)[2][0]
        cv2.rectangle(mask, (x2, y2), (x1-50, y1-50), (0, 0, 0), thickness=cv2.FILLED)

        image = cv2.bitwise_and(mask, image)
        # cv2.imshow("  ", image)
        # cv2.waitKey(0)

    image = cv2.imread(image_path)
    for ghost_corners in ghosts:
        cv2.polylines(image, [np.int32(ghost_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

    return image

ghost_paths = ['scary_ghost.png', 'candy_ghost.png', 'pampkin_ghost.png']  # Пути к изображениям призраков
result_images = find_and_highlight_ghosts('lab7.png', ghost_paths)

cv2.imshow(f'Result ', result_images)

cv2.waitKey(0)
cv2.destroyAllWindows()
