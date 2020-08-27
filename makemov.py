import cv2
import os

image_folder = 'results_large'
video_name = 'video_large.avi'

images = [os.path.join(image_folder, 'im' + str(i) + '.png') for i in range(1690)]
frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
