# SLICsuperpixels
Final project for course on computer vision. [SLIC superpixels paper](http://www.kev-smith.com/papers/SLIC_Superpixels.pdf). I found this concept novel, notably for uses in medicine (specifically in medical imaging) but also artistic. Initially I hoped implementing the algorithm and instead of drawing lines indicating the different super pixels, using a Gaussian blur on the seperation lines would create an interesting effect. However, this experiment was underwhelming and excluded from this repository.

##### To run project:

slic.py Lenna.png 1000 40

Where:
 arg[1] is the name of the image you would like to run the program on. I have provided Lenna.png in the folder already, but the algorithm should run on any aspect ratio and image.

 arg[2] is the number of pixels you would like the program to create. Different (less impressive) results occur when you reduce the number of pixels you want. And likewise, when you request a higher pixel total, 10,000 for example, the results are great, but the runtime is not.

 	recommended: 100, 1,000, or 10,000

 arg[3] is SLIC_m which is described in the paper as the the control in compactness of a superpixel. They use 10, but I found that tuning this parameter was most key in finding the best (most accurate) superpixels.

 	recommended (for Lenna.png) 40
