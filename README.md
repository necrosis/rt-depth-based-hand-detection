# rt-depth-based-hand-detection
Real time depth based hand detection
############

Based on this article: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3967598/
And also on this repo: https://github.com/Simon-Hohberg/Viola-Jones 

Includes
=======
Python(3.6) source of lib, trainer and tester, a lot of bugs and optimization issues. Basicly, it works with selfmade depth image format and info files (I didn't look at available formats, sorry). I'll add setup.py in future.


How to use
==========
To create a cascade, run this:
	trainer.py -p <positive file> -n <negative file> -num <number of week classifiers> -o <output json file with cascade> -d <detection rate (def 0.9)> -e <error rate (def 0.05)> -x <list of Nx/Ny values, which will combined (example: 2 4 8 12)> [-imgdir <prefix to all files in postivie/negative files>]

To test the created cascade, run this:
	tester.py -c <path to cascade .json file> -t <path to test file (in spec format)> [-imgdir <prefix to all files in postivie/negative files>]

Pos|Neg files format
==========
Positive/Negative files are files in format:
<image_name> 1 <x> <y> <w> <h>
Where x,y,w,h - rectanguler region of interest in image image_name.
Also, there are some functions to load images without this files.

Raw file format
==========
File image_name is a .raw format, witch structure:
Header, image_width, image_height, RAW_DATA
0xD(16)    (32)          (32)      16*(image_width * image_height)

Test file format
==========
<image_name> <0/1> <x> <y> <w> <h>
<0/1>: 0 - is negative region, 1 - positive region. x,y,w,h - rectanguler region of interest in image image_name.

