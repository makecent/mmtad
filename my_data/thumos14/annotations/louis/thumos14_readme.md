# Introduction
THUMOS14 collects sports videos from 20 classes.
It contains 2756 trimmed videos for training, 200 and 213 (212 valid, see below A.2) untrimmed videos for validation and testing, respectively.
There are 2756, 3096 (3007 are not ambiguous) and 3454 (3358 are not ambiguous) action instances for training, validation and testing, respectively.
The average duration of videos and actions in Untrimmed videos are 4.4 minutes and 5 seconds, respectively.
mAP@[0.3:0.1:0.7] and their mean are often used to benchmark the performance.

# Additional information
1. Note that the FPS of videos are not consistent: most are 30 but some are around 25.
2. Note that the number of extracted frames could be inconsistent with the num_frames in VideoReader.
But this only happens to the videos in training split. It seems because the videos in validation and testing are all .mp4
which do NOT have the inconsistent number of frames problem. In contrast, there are lots of .avi videos in training split
that have the problem.

# Details about the annotations file I created.
1. The label indexes I used are as below:
 classes = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
           'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
           'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
           'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
           'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
           'VolleyballSpiking')
 'Ambiguous' is labeled as -1.

2. There is one test video named "video_test_0001292" that only has 'Ambiguous' actions annotated, thereby is ignored.
3. All 'CliffDiving' actions are also annotated as 'Diving' actions. I did NOT handle this as multi-label classification
but simply adding two annotations that have same intervals but different labels, for both training and testing.
4. video_validation_0000364 should be corrected, the last two annotations are out of range (greater than video duration).
   video_validation_0000856 should be corrected, the last two annotations are out of range.
   video_test_0000814 should be corrected, the last three annotations are out of range.
   video_test_0001081 should be corrected, the last one annotation is out of range.
   *I found the remaining annotations in these videos to be correct upon manual inspection 
5. In validation set, there are (47, 16, 1, 0) actions shorter (<=) than (1.0, 0.5, 0.2, 0.1) seconds, respectively.
   In testing set, there are (50, 14, 0, 0) actions shorter (<=) than (1.0, 0.5, 0.2, 0.1) seconds, respectively.
   They were NOT handled and all kept as they are in my annotation file.

6. video_validation_0000176[42:44] are strange, two segments have same start, different end, and different labels (5, 6)
   video_validation_0000184[13:15] are strange, two segments have same start, different end, and different labels (5, 6)
   They were NOT handled and all kept as they are in my annotation file.
7. video_test_0000270 may be wrong, most annotations are out of range. After manual checking, the whole annotation (29) should be dropped.
8. video_test_0001496 was found to be wrong annotated by accident. The whole annotation (27) should be dropped.
9. **These wrong annotations are all KEPT in my annotations file**, and you are encouraged to handle them during the loading. 
While the testing annotations should be kept if you want a fair comparison with other work.
