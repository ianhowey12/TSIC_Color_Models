# TSIC_Color_Models
An original project I conducted which examines the effects of different color models on convolutional neural networks (CNNs) and their ability to classify images of traffic signs through a dedicated experimental procedure.

This project's code implements the algorithm and its associated combined training and testing procedure used for the study I conducted.
The project examines the performance of the RGB, RGBK, CMY, CMYK, HSV, HSL, K grayscale, and L grayscale color models when used in conjunction with a CNN algorithm of varying parameters. Performance is determined using a sample of real images of traffic signs from the CURE-TSR image dataset. Classification accuracy, classification duration, and memory usage are the dependent variables. The program conducts the full experimental process for the selected image sample, which is customizable. The project aims to discover the value of alternative color models for this task and to determine the impact color data has on CNN algorithm performance and general functioning for similar computer vision tasks. Applications for such algorithm models focus on the development of autonomous vehicles in the context of traffic sign recognition, although for other target objects, many diverse settings require intelligent computer vision processes and benefit from algorithm design and development.

The CURE-TSR image dataset involved in this study can be downloaded at https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition

This very large and high-quality dataset has been made public by researchers at the Georgia Institute of Technology. Their repository can be found here: https://github.com/olivesgatech/CURE-TSR


Citations for this dataset:

D. Temel, G. Kwon, M. Prabhushankar, and G. AlRegib, "CURE-TSR: Challenging unreal and real environments for traffic sign recognition," in Neural Information Processing Systems (NIPS) Workshop on Machine Learning for Intelligent Transportation Systems, Long Beach, U.S., December 2017, https://arxiv.org/abs/1712.02463

D. Temel, T. Alshawi, M-H Chen and G. AlRegib, "Challenging Environments for Traffic Sign Detection: Reliability Assessment under Inclement Conditions," February 2019, https://arxiv.org/abs/1902.06857

D. Temel, M-H Chen, and G. AlRegib, "Traffic Sign Detection Under Challenging Conditions: A Deeper Look into Performance Variations and Spectral Characteristics," February 2019, https://ieeexplore.ieee.org/document/8793235

D. Temel and G. AlRegib, "Traffic Signs in the Wild: Highlights from the IEEE Video and Image Processing Cup 2017 Student Competition [SP Competitions]," in IEEE Signal Processing Magazine, vol. 35, no. 2, pp. 154-161, March 2018.
