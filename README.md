# Opencv_HOOFR

This is the version of Opencv 3.1 which includes the research about HOOFR algorithm. The use of this algorithm is similar to others like ORB, BRISK,.... It has also the version of HOOFR cuda implementaion but it is built with no OpenCL support.

#Declaration: 
Ptr<HOOFR> hoofr_al = cv::HOOFR::create(..,..,..,..);
#Call: 
hoofr_al->detect(); or hoofr_al->compute(); or hoofr_al->DetectandCompute(); 
