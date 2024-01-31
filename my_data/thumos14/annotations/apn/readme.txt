apn_train has two annotation files, one is used for raw frames training and the other is used for raw videos training. 
This is because the number frames are different between 'rawframes' (extracted frames) and 'video' (frame_num by cv2.Videocapture).

found some annotations may be wrong(2022 Aug 17) but I did NOT removed them:
video_test_0000045.mp4,7059,3924,3948,15
video_test_0000045.mp4,7059,3888,3909,15
video_test_0000045.mp4,7059,6768,6855,15
