import cv2 as cv


def main():
    cap = cv.VideoCapture("data/d4ff4597177c94e6548ea43dd0bdee7f.mp4")
    roadhog = cv.imread("images/roadhog.png")
    h, w, _ = roadhog.shape

    frame = 0

    while frame < 9 * 30 * 60:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame)

        ret, img = cap.read()
        res = cv.matchTemplate(img, roadhog, method=cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val > 0.95:
            print(f"Found a Roadhog kill at frame {frame}")
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            final_frame = cv.rectangle(img, top_left, bottom_right, 255, 2)
            cv.imwrite(f"output/kill-at-{frame}.png", final_frame)
        print(f"Processed frame {frame}")
        frame += 30

    cap.release()


if __name__ == "__main__":
    main()
