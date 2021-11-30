import glob
import os
import sys

import cv2 as cv


def video_sampling(video_path, store_path, total_frames, frame_nums):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(-1)
    for i in range(0, total_frames, int(total_frames / frame_nums)):
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        else:
            cv.imwrite(os.path.join(store_path, 'step' + str(int(i / (total_frames / frame_nums) + 1)) + '.jpg'), frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':

    # total_frames = 300
    # frame_nums = 50

    participants = glob.glob('E:\\yan0\\bishe_test\\Test\\*')
    for participant in participants:
        print(participant)
        video_list = glob.glob(os.path.join(participant, '*'))
        for video in video_list:
            print(video)
            video_sampling(glob.glob(os.path.join(video, '*.avi'))[0], video, 300, 30)
