/*
The CURE-TSR traffic sign image dataset can be downloaded at https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition.
Before using, extract each of the 61 sub-folders within the "Real_Train" folder of the dataset to a folder named "Train" on Windows C: drive.
This extraction process should take 1-2 minutes per sub-folder or 1-2 hours in total, depending on the computer being used.
The test functions inside main() at the bottom of this file can be used to test the program by removing "//" before each test function's name.
Remove "//" before initializeImagesAll(); or before initializeImagesChallengeFree(); or before initializeImagesLowChallenge(); inside main() depending on the subset of images to sample.
As it is currently set up, the program will run the full experimental process using runTest().
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int imageCounts[14] = { 1087, 290, 85, 38, 807, 89, 1091, 206, 104, 193, 80, 116, 626, 2478 };
// most folders have 2480 parking signs, yet LensBlur-1, Darkening-1, and GaussianBlur-1 have 2478, thus, 2478 is used

// number of images
#define numPerSign 38 // per sign within each folder
#define numPerFolder 532 // within each folder
#define numPerCondition 2660 // per visual condition
#define numTotal 32452 // in sample
#define numWithChallenges 31920 // in sample excluding ChallengeFree sub-folder
#define numTraining 24339 // in training set, 3/4 of total sample

// image file address being used
char* currentAddress = NULL;

// images left of each sign within each folder to choose from for randomization
char imagesLeft[2500];

// properties of sampled images before randomization
char orderedConditions[numTotal];
char orderedChallenges[numTotal];
char orderedSigns[numTotal];
int orderedNumbers[numTotal];

// properties of sampled images in randomized order
char imageConditions[numTotal];
char imageChallenges[numTotal];
char imageSigns[numTotal];
int imageNumbers[numTotal];

// number of convolutional filters in current test
#define maxNumFilters 96
int numFilters = 0;
int numFiltersPerColor = 0;

// convolutional filters
#define maxFilterArea 121
float filter[maxNumFilters][maxFilterArea];

// selected filter during feature extraction
float* F;

// selected filter number during feature extraction
int f;

// i-value (height) and j-value (width) of the maximum value on the feature map, used for training
int filterMapMaxI[maxNumFilters];
int filterMapMaxJ[maxNumFilters];

// maximum values on the feature maps, used as the neural network input for each filter
float nnInputs[maxNumFilters];
float nnInputTotals[maxNumFilters];

// neural network weight values used to compute the outputs given the inputs
float nnWeights1[maxNumFilters][maxNumFilters];
float nnWeights2[14][maxNumFilters];

// neural network bias values used to compute the outputs given the inputs
float nnBiases1[maxNumFilters];
float nnBiases2[14];

// values in middle layer of neural network
float nnHidden[maxNumFilters];

// predictive values for each traffic sign type, outputted by the neural network
float nnOutputs[14];

// width, height, and number of pixels of the selected image
int innerWidth;
int innerHeight;
int width;
int height;
int numPixels;
int lineLength;

// convolutional filter size
char filterSize;
char filterArea;
char halfFilterArea;
char padding;
char doublePadding;

#define maxImageSize 80000

// image pixel colors
unsigned char r[maxImageSize];
unsigned char g[maxImageSize];
unsigned char b[maxImageSize];

unsigned char c1[maxImageSize];
unsigned char c2[maxImageSize];
unsigned char c3[maxImageSize];
unsigned char c4[maxImageSize];

char currentColorModel = 0;

// stats for testing
int imageNumber = 0;
int imagesCorrect = 0;
int imagesClassified = 0;
int timeTraining = 0;
int timeTesting = 0;
int timeTotal = 0;

// memory usage measured during each test
int memoryUsage = 0;

// data from image files stored as a text string
#define maxFileSize 240000
unsigned char file[maxFileSize];

// clock start and stop millisecond values used for timing
int start;
int stop;

// file address templates
char a0[] = "C:\\Train\\ChallengeFree\\01_  _  _  _    .bmp";
char a1[] = "C:\\Train\\Decolorization- \\01_  _  _  _    .bmp";
char a2[] = "C:\\Train\\LensBlur- \\01_  _  _  _    .bmp";
char a3[] = "C:\\Train\\CodecError- \\01_  _  _  _    .bmp";
char a4[] = "C:\\Train\\Darkening- \\01_  _  _  _    .bmp";
char a5[] = "C:\\Train\\DirtyLens- \\01_  _  _  _    .bmp";
char a6[] = "C:\\Train\\Exposure- \\01_  _  _  _    .bmp";
char a7[] = "C:\\Train\\GaussianBlur- \\01_  _  _  _    .bmp";
char a8[] = "C:\\Train\\Noise- \\01_  _  _  _    .bmp";
char a9[] = "C:\\Train\\Rain- \\01_  _  _  _    .bmp";
char a10[] = "C:\\Train\\Shadow- \\01_  _  _  _    .bmp";
char a11[] = "C:\\Train\\Snow- \\01_  _  _  _    .bmp";
char a12[] = "C:\\Train\\Haze- \\01_  _  _  _    .bmp";

// get a random decimal number with av as the mean value and range as the possible range of values
float randFloat(float min, float range) {
	return min + (((float)rand()) / 32768.0f) * range;
}

// get a random decimal number with min as the minimun possible value and range as the possible range of values
int randInt(int min, int range) {
	return min + (((unsigned int)rand()) % range);
}

// fill arrays with zeros
void setup() {

	for (int i = 0; i < maxFileSize; i++) {
		file[i] = 0;
	}

	for (int i = 0; i < maxNumFilters; i++) {
		nnInputs[i] = 0.0f;
		nnHidden[i] = 0.0f;
		nnBiases1[i] = 0.0f;
		for (int j = 0; j < maxNumFilters; j++) {
			nnWeights1[i][j] = 0.0f;
		}
	}

	for (int i = 0; i < 14; i++) {
		nnOutputs[i] = 0.0f;
		nnBiases2[i] = 0.0f;
		for (int j = 0; j < maxNumFilters; j++) {
			nnWeights2[i][j] = 0.0f;
		}
	}

	for (int i = 0; i < maxImageSize; i++) {
		r[i] = 0;
		g[i] = 0;
		b[i] = 0;

		c1[i] = 0;
		c2[i] = 0;
		c3[i] = 0;
		c4[i] = 0;
	}

	for (int i = 0; i < maxNumFilters; i++) {
		for (int j = 0; j < maxFilterArea; j++) {
			filter[i][j] = 0.0f;
		}
	}
}

// get number of colors in the current color model
char getNumColors() {
	if (currentColorModel == 1 || currentColorModel == 3) {
		return 4;
	}
	else {
		if (currentColorModel > 5) {
			return 1;
		}
	}
	return 3;
}

// selects 32,452 random images as the training and testing sample
void initializeImages() {

	// images left in each folder for randomization
	int numImagesLeft = 0;
	int count = 0;
	int r = 0;

	for (int i = 0; i < numPerFolder; i++) {
		orderedConditions[i] = 0;
		orderedChallenges[i] = 0;
	}

	for (int i = 0; i < numTotal; i++) {
		orderedSigns[i] = ((i / numPerSign) % 14) + 1;
	}

	int added = 0;
	// randomly choose images within each sign in each sub-folder
	for (int i = 0; i < 854; i++) {
		added = i * numPerSign;
		count = imageCounts[i % 14];
		for (int j = 0; j < count; j++) {
			imagesLeft[j] = 0;
		}

		for (int j = 0; j < numPerSign; j++) {
			r = randInt(1, count);
			while (imagesLeft[r - 1]) {
				r = randInt(1, count);
			}

			imagesLeft[r - 1] = 1;
			orderedNumbers[added + j] = r;
		}
	}

	numImagesLeft = numTotal;

	// arrange the sample images in random order
	for (int i = 0; i < numTotal; i++) {
		r = randInt(0, numImagesLeft);
		imageConditions[i] = orderedConditions[r];
		imageChallenges[i] = orderedChallenges[r];
		imageSigns[i] = orderedSigns[r];
		imageNumbers[i] = orderedNumbers[r];

		numImagesLeft--;
		for (int j = r; j < numImagesLeft; j++) {
			orderedConditions[j] = orderedConditions[j + 1];
			orderedChallenges[j] = orderedChallenges[j + 1];
			orderedSigns[j] = orderedSigns[j + 1];
			orderedNumbers[j] = orderedNumbers[j + 1];
		}
	}
}

// use all image visual conditions as the sample
void initializeImagesAll() {
	for (int i = 0; i < numWithChallenges; i++) {
		orderedConditions[i + numPerFolder] = (i / numPerCondition) + 1;
		orderedChallenges[i + numPerFolder] = ((i / numPerFolder) % 5) + 1;
	}
	initializeImages();
}

// use only challenge-free images as the sample
void initializeImagesChallengeFree() {
	for (int i = 0; i < numWithChallenges; i++) {
		orderedConditions[i + numPerFolder] = 0;
		orderedChallenges[i + numPerFolder] = 0;
	}
	initializeImages();
}

// use only challenge level 1 for images with challenge levels in the sample
void initializeImagesLowChallenge() {
	for (int i = 0; i < numWithChallenges; i++) {
		orderedConditions[i + numPerFolder] = (i / numPerCondition) + 1;
		orderedChallenges[i + numPerFolder] = 1;
	}
	initializeImages();
}

// modify one of the file address templates above for one particular image
char* modifyAddress(char* a, char s, char condition, char challenge, char sign, int number) {
	a[s - 5] = challenge + 48;
	a[s] = (sign / 10) + 48;
	a[s + 1] = (sign % 10) + 48;
	a[s + 3] = (condition / 10) + 48;
	a[s + 4] = (condition % 10) + 48;
	a[s + 6] = (challenge / 10) + 48;
	a[s + 7] = (challenge % 10) + 48;
	a[s + 9] = (number / 1000) + 48;
	a[s + 10] = ((number / 100) % 10) + 48;
	a[s + 11] = ((number / 10) % 10) + 48;
	a[s + 12] = (number % 10) + 48;
	return a;
}

// create the image file address from the visual condition, challenge level, sign type, and image number
char* getAddress(char condition, char challenge, char sign, int number) {

	if (condition == 0) {
		challenge = 0;
	}
	
	switch (condition) {
	case 0:
	{
		a0[26] = (sign / 10) + 48;
		a0[26 + 1] = (sign % 10) + 48;
		a0[26 + 3] = (condition / 10) + 48;
		a0[26 + 4] = (condition % 10) + 48;
		a0[26 + 6] = (challenge / 10) + 48;
		a0[26 + 7] = (challenge % 10) + 48;
		a0[26 + 9] = (number / 1000) + 48;
		a0[26 + 10] = ((number / 100) % 10) + 48;
		a0[26 + 11] = ((number / 10) % 10) + 48;
		a0[26 + 12] = (number % 10) + 48;
		return a0;
	}
	case 1: return modifyAddress(a1, 29, condition, challenge, sign, number);
	case 2: return modifyAddress(a2, 23, condition, challenge, sign, number);
	case 3: return modifyAddress(a3, 25, condition, challenge, sign, number);
	case 4: return modifyAddress(a4, 24, condition, challenge, sign, number);
	case 5: return modifyAddress(a5, 24, condition, challenge, sign, number);
	case 6: return modifyAddress(a6, 23, condition, challenge, sign, number);
	case 7: return modifyAddress(a7, 27, condition, challenge, sign, number);
	case 8: return modifyAddress(a8, 20, condition, challenge, sign, number);
	case 9: return modifyAddress(a9, 19, condition, challenge, sign, number);
	case 10: return modifyAddress(a10, 21, condition, challenge, sign, number);
	case 11: return modifyAddress(a11, 19, condition, challenge, sign, number);
	case 12: return modifyAddress(a12, 19, condition, challenge, sign, number);
	}
	return (char*)0;
}

// convert RGB pixel color array to another color model
void convert() {

	unsigned char max = 0;
	unsigned char min = 0;

	float delta = 0.0f;
	int sum = 0;

	switch (currentColorModel) {
	case 0:
	case 1:
		// For RGB, set c1 to R, set c2 to G, set c3 to B; for RGBK, set c4 to calculated K values
		for (int i = 0; i < numPixels; i++) {

			c1[i] = r[i];
			c2[i] = g[i];
			c3[i] = b[i];

			if (currentColorModel == 1) {
				max = r[i];
				if (g[i] > max) { max = g[i]; }
				if (b[i] > max) { max = b[i]; }
				c4[i] = 255 - max;
			}
		}
		break;
	case 2:
	case 3:
		// For CMY, set c1 to calculated C values, set c2 to calculated M values, set c3 to calculated Y values; for CMYK, set c4 to calculated K values
		for (int i = 0; i < numPixels; i++) {

			max = r[i];
			if (g[i] > max) { max = g[i]; }
			if (b[i] > max) { max = b[i]; }
			c1[i] = (int)(255.999f * (float)(max - r[i]) / (float)max);
			c2[i] = (int)(255.999f * (float)(max - g[i]) / (float)max);
			c3[i] = (int)(255.999f * (float)(max - b[i]) / (float)max);
			if (currentColorModel == 3) {
				c4[i] = 255 - max;
			}
		}
		break;
	case 4:
	case 5:
		// For HSV, set c1 to calculated H values, set c2 to calculated S values, set c3 to calculated V values; for HSL, set c3 to calculated L values
		for (int i = 0; i < numPixels; i++) {

			max = r[i];
			if (g[i] > max) { max = g[i]; }
			if (b[i] > max) { max = b[i]; }

			min = r[i];
			if (g[i] < min) { min = g[i]; }
			if (b[i] < min) { min = b[i]; }

			delta = (float)((int)max - (int)min);

			if (r[i] >= g[i] && r[i] >= b[i]) {
				c1[i] = (int)(42.667f * ((float)((int)g[i] - (int)b[i]) / delta)) % 256;
			}
			else {
				if (g[i] >= b[i] && g[i] >= r[i]) {
					c1[i] = (int)(42.667f * ((float)((int)b[i] - (int)r[i]) / delta) + 85.333f) % 256;
				}
				else {
					c1[i] = (int)(42.667f * ((float)((int)r[i] - (int)g[i]) / delta) + 170.667f) % 256;
				}
			}

			c2[i] = (int)(255.999f * delta / max);

			if (currentColorModel == 4) {
				c3[i] = max;
			}
			else {
				sum = (int)max + (int)min;
				c3[i] = (unsigned char)(sum / 2);
			}
		}
		break;
	case 6:
	case 7:
		// For K grayscale, set c1 to calculated K values; for L grayscale, set c1 to calculated L values
		for (int i = 0; i < numPixels; i++) {

			max = r[i];
			if (g[i] > max) { max = g[i]; }
			if (b[i] > max) { max = b[i]; }

			min = r[i];
			if (g[i] < min) { min = g[i]; }
			if (b[i] < min) { min = b[i]; }

			if (currentColorModel == 6) {
				c1[i] = 255 - max;
			}
			else {
				sum = (int)max + (int)min;
				c1[i] = (unsigned char)(sum / 2);
			}
		}
		break;
	}
}

// read an image file given the image's file address, constructing RGB arrays for the image with padding and filling them with the pixel color data
void readFile(char* address) {

	FILE* fp;
	fopen_s(&fp, address, "r");
	if (fp == NULL) {
		printf("Couldn't open file %s\n", address);
	}
	else {
		fread(file, sizeof(char), maxFileSize, fp);
		fclose(fp);
	}

	// reading width and height
	innerWidth = file[18];
	innerHeight = file[22];
	width = innerWidth + doublePadding;
	height = innerHeight + doublePadding;

	numPixels = width * height;

	lineLength = innerWidth * 3 + (innerWidth % 4);

	// structuring the color data into three arrays (RGB); transferring each pixel color value from the string of text from the file to these arrays
	for (int i = 0; i < innerHeight; i++) {
		for (int j = 0; j < innerWidth; j++) {
			r[(height - i - padding - 1) * width + j + padding] = file[i * lineLength + j * 3 + 56];
			g[(height - i - padding - 1) * width + j + padding] = file[i * lineLength + j * 3 + 55];
			b[(height - i - padding - 1) * width + j + padding] = file[i * lineLength + j * 3 + 54];
		}
	}

	// surrounding the pixel color data with 0s as padding for use in feature extraction
	for (int p = 0; p < padding; p++) {
		for (int i = 1; i <= height; i++) {
			r[(i - 1) * width + p] = 0;
			g[(i - 1) * width + p] = 0;
			b[(i - 1) * width + p] = 0;
			r[i * width - 1 - p] = 0;
			g[i * width - 1 - p] = 0;
			b[i * width - 1 - p] = 0;
		}
		for (int i = 0; i < width; i++) {
			r[p * width + i] = 0;
			g[p * width + i] = 0;
			b[p * width + i] = 0;
			r[(height - 1 - p) * width + i] = 0;
			g[(height - 1 - p) * width + i] = 0;
			b[(height - 1 - p) * width + i] = 0;
		}
	}
}

// fill all convolutional filters, neural network weights, and neural network biases with values prior to training in each trial
void randomizeParameters() {
	for (int i = 0; i < maxNumFilters; i++) {
		for (int j = 0; j < maxFilterArea; j++) {
			filter[i][j] = randFloat(-5.0f, 10.0f);
		}
		for (int j = 0; j < maxNumFilters; j++) {
			nnWeights1[j][i] = randFloat(-0.5f, 1.0f);
		}
		for (int j = 0; j < 14; j++) {
			nnWeights2[j][i] = randFloat(-0.5f, 1.0f);
		}
		nnBiases1[i] = 0.0f;
	}
	for (int i = 0; i < 14; i++) {
		nnBiases2[i] = 0.0f;
	}
	for (int i = 0; i < maxNumFilters; i++) {
		nnInputTotals[i] = 0.0f;
	}
}

// compute feature map for one convolutional filter and one image pixel color array
void convolveColor(unsigned char* a) {
	float total = 0;
	switch (filterSize) {
	case 3:
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				total = F[0] * (float)a[(i - 1) * width + j - 1] + F[1] * (float)a[(i - 1) * width + j] + F[2] * (float)a[(i - 1) * width + j + 1] +
				F[3] * (float)a[i * width + j - 1] + F[4] * (float)a[i * width + j] + F[5] * (float)a[i * width + j + 1] +
				F[6] * (float)a[(i + 1) * width + j - 1] + F[7] * (float)a[(i + 1) * width + j] + F[8] * (float)a[(i + 1) * width + j + 1];
				if (total > nnInputs[f]) {
					nnInputs[f] = total;
					filterMapMaxI[f] = i;
					filterMapMaxJ[f] = j;
				}
			}
		}
		break;
	case 5:
		for (int i = 2; i < height - 2; i++) {
			for (int j = 2; j < width - 2; j++) {
				total = F[0] * (float)a[(i - 2) * width + j - 2] + F[1] * (float)a[(i - 2) * width + j - 1] + F[2] * (float)a[(i - 2) * width + j] +
				F[3] * (float)a[(i - 2) * width + j + 1] + F[4] * (float)a[(i - 2) * width + j + 2] + F[5] * (float)a[(i - 1) * width + j - 2] +
				F[6] * (float)a[(i - 1) * width + j - 1] + F[7] * (float)a[(i - 1) * width + j] + F[8] * (float)a[(i - 1) * width + j + 1] +
				F[9] * (float)a[(i - 1) * width + j + 2] + F[10] * (float)a[i * width + j - 2] + F[11] * (float)a[i * width + j - 1] +
				F[12] * (float)a[i * width + j] + F[13] * (float)a[i * width + j + 1] + F[14] * (float)a[i * width + j + 2] +
				F[15] * (float)a[(i + 1) * width + j - 2] + F[16] * (float)a[(i + 1) * width + j - 1] + F[17] * (float)a[(i + 1) * width + j] +
				F[18] * (float)a[(i + 1) * width + j + 1] + F[19] * (float)a[(i + 1) * width + j + 2] + F[20] * (float)a[(i + 2) * width + j - 2] +
				F[21] * (float)a[(i + 2) * width + j - 1] + F[22] * (float)a[(i + 2) * width + j] + F[23] * (float)a[(i + 2) * width + j + 1] +
				F[24] * (float)a[(i + 2) * width + j + 2];
				if (total > nnInputs[f]) {
					nnInputs[f] = total;
					filterMapMaxI[f] = i;
					filterMapMaxJ[f] = j;
				}
			}
		}
		break;
	case 7:
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				total = F[0] * (float)a[(i - 3) * width + j - 3] + F[1] * (float)a[(i - 3) * width + j - 2] + F[2] * (float)a[(i - 3) * width + j - 1] +
					F[3] * (float)a[(i - 3) * width + j] + F[4] * (float)a[(i - 3) * width + j + 1] + F[5] * (float)a[(i - 3) * width + j + 2] +
					F[6] * (float)a[(i - 3) * width + j + 3] + F[7] * (float)a[(i - 2) * width + j - 3] + F[8] * (float)a[(i - 2) * width + j - 2] +
					F[9] * (float)a[(i - 2) * width + j - 1] + F[10] * (float)a[(i - 2) * width + j] + F[11] * (float)a[(i - 2) * width + j + 1] +
					F[12] * (float)a[(i - 2) * width + j + 2] + F[13] * (float)a[(i - 2) * width + j + 3] + F[14] * (float)a[(i - 1) * width + j - 3] +
					F[15] * (float)a[(i - 1) * width + j - 2] + F[16] * (float)a[(i - 1) * width + j - 1] + F[17] * (float)a[(i - 1) * width + j] +
					F[18] * (float)a[(i - 1) * width + j + 1] + F[19] * (float)a[(i - 1) * width + j + 2] + F[20] * (float)a[(i - 1) * width + j + 3] +
					F[21] * (float)a[i * width + j - 3] + F[22] * (float)a[i * width + j - 2] + F[23] * (float)a[i * width + j - 1] +
					F[24] * (float)a[i * width + j] + F[25] * (float)a[i * width + j + 1] + F[26] * (float)a[i * width + j + 2] +
					F[27] * (float)a[i * width + j + 3] + F[28] * (float)a[(i + 1) * width + j - 3] + F[29] * (float)a[(i + 1) * width + j - 2] +
					F[30] * (float)a[(i + 1) * width + j - 1] + F[31] * (float)a[(i + 1) * width + j] + F[32] * (float)a[(i + 1) * width + j + 1] +
					F[33] * (float)a[(i + 1) * width + j + 2] + F[34] * (float)a[(i + 1) * width + j + 3] + F[35] * (float)a[(i + 2) * width + j - 3] +
					F[36] * (float)a[(i + 2) * width + j - 2] + F[37] * (float)a[(i + 2) * width + j - 1] + F[38] * (float)a[(i + 2) * width + j] +
					F[39] * (float)a[(i + 2) * width + j + 1] + F[40] * (float)a[(i + 2) * width + j + 2] + F[41] * (float)a[(i + 2) * width + j + 3] +
					F[42] * (float)a[(i + 3) * width + j - 3] + F[43] * (float)a[(i + 3) * width + j - 2] + F[44] * (float)a[(i + 3) * width + j - 1] +
					F[45] * (float)a[(i + 3) * width + j] + F[46] * (float)a[(i + 3) * width + j + 1] + F[47] * (float)a[(i + 3) * width + j + 2] +
					F[48] * (float)a[(i + 3) * width + j + 3];
				if (total > nnInputs[f]) {
					nnInputs[f] = total;
					filterMapMaxI[f] = i;
					filterMapMaxJ[f] = j;
				}
			}
		}
		break;
	case 9:
		for (int i = 4; i < height - 4; i++) {
			for (int j = 4; j < width - 4; j++) {
				total = F[0] * (float)a[(i - 4) * width + j - 4] + F[1] * (float)a[(i - 4) * width + j - 3] + F[2] * (float)a[(i - 4) * width + j - 2] +
					F[3] * (float)a[(i - 4) * width + j - 1] + F[4] * (float)a[(i - 4) * width + j] + F[5] * (float)a[(i - 4) * width + j + 1] +
					F[6] * (float)a[(i - 4) * width + j + 2] + F[7] * (float)a[(i - 4) * width + j + 3] + F[8] * (float)a[(i - 4) * width + j + 4] +
					F[9] * (float)a[(i - 3) * width + j - 4] + F[10] * (float)a[(i - 3) * width + j - 3] + F[11] * (float)a[(i - 3) * width + j - 2] +
					F[12] * (float)a[(i - 3) * width + j - 1] + F[13] * (float)a[(i - 3) * width + j] + F[14] * (float)a[(i - 3) * width + j + 1] +
					F[15] * (float)a[(i - 3) * width + j + 2] + F[16] * (float)a[(i - 3) * width + j + 3] + F[17] * (float)a[(i - 3) * width + j + 4] +
					F[18] * (float)a[(i - 2) * width + j - 4] + F[19] * (float)a[(i - 2) * width + j - 3] + F[20] * (float)a[(i - 2) * width + j - 2] +
					F[21] * (float)a[(i - 2) * width + j - 1] + F[22] * (float)a[(i - 2) * width + j] + F[23] * (float)a[(i - 2) * width + j + 1] +
					F[24] * (float)a[(i - 2) * width + j + 2] + F[25] * (float)a[(i - 2) * width + j + 3] + F[26] * (float)a[(i - 2) * width + j + 4] +
					F[27] * (float)a[(i - 1) * width + j - 4] + F[28] * (float)a[(i - 1) * width + j - 3] + F[29] * (float)a[(i - 1) * width + j - 2] +
					F[30] * (float)a[(i - 1) * width + j - 1] + F[31] * (float)a[(i - 1) * width + j] + F[32] * (float)a[(i - 1) * width + j + 1] +
					F[33] * (float)a[(i - 1) * width + j + 2] + F[34] * (float)a[(i - 1) * width + j + 3] + F[35] * (float)a[(i - 1) * width + j + 4] +
					F[36] * (float)a[i * width + j - 4] + F[37] * (float)a[i * width + j - 3] + F[38] * (float)a[i * width + j - 2] +
					F[39] * (float)a[i * width + j - 1] + F[40] * (float)a[i * width + j] + F[41] * (float)a[i * width + j + 1] +
					F[42] * (float)a[i * width + j + 2] + F[43] * (float)a[i * width + j + 3] + F[44] * (float)a[i * width + j + 4] +
					F[45] * (float)a[(i + 1) * width + j - 4] + F[46] * (float)a[(i + 1) * width + j - 3] + F[47] * (float)a[(i + 1) * width + j - 2] +
					F[48] * (float)a[(i + 1) * width + j - 1] + F[49] * (float)a[(i + 1) * width + j] + F[50] * (float)a[(i + 1) * width + j + 1] +
					F[51] * (float)a[(i + 1) * width + j + 2] + F[52] * (float)a[(i + 1) * width + j + 3] + F[53] * (float)a[(i + 1) * width + j + 4] +
					F[54] * (float)a[(i + 2) * width + j - 4] + F[55] * (float)a[(i + 2) * width + j - 3] + F[56] * (float)a[(i + 2) * width + j - 2] +
					F[57] * (float)a[(i + 2) * width + j - 1] + F[58] * (float)a[(i + 2) * width + j] + F[59] * (float)a[(i + 2) * width + j + 1] +
					F[60] * (float)a[(i + 2) * width + j + 2] + F[61] * (float)a[(i + 2) * width + j + 3] + F[62] * (float)a[(i + 2) * width + j + 4] +
					F[63] * (float)a[(i + 3) * width + j - 4] + F[64] * (float)a[(i + 3) * width + j - 3] + F[65] * (float)a[(i + 3) * width + j - 2] +
					F[66] * (float)a[(i + 3) * width + j - 1] + F[67] * (float)a[(i + 3) * width + j] + F[68] * (float)a[(i + 3) * width + j + 1] +
					F[69] * (float)a[(i + 3) * width + j + 2] + F[70] * (float)a[(i + 3) * width + j + 3] + F[71] * (float)a[(i + 3) * width + j + 4] +
					F[72] * (float)a[(i + 4) * width + j - 4] + F[73] * (float)a[(i + 4) * width + j - 3] + F[74] * (float)a[(i + 4) * width + j - 2] +
					F[75] * (float)a[(i + 4) * width + j - 1] + F[76] * (float)a[(i + 4) * width + j] + F[77] * (float)a[(i + 4) * width + j + 1] +
					F[78] * (float)a[(i + 4) * width + j + 2] + F[79] * (float)a[(i + 4) * width + j + 3] + F[80] * (float)a[(i + 4) * width + j + 4];
				if (total > nnInputs[f]) {
					nnInputs[f] = total;
					filterMapMaxI[f] = i;
					filterMapMaxJ[f] = j;
				}
			}
		}
		break;
	case 11:
		for (int i = 5; i < height - 5; i++) {
			for (int j = 5; j < width - 5; j++) {
				total = F[0] * (float)a[(i - 5) * width + j - 5] + F[1] * (float)a[(i - 5) * width + j - 4] + F[2] * (float)a[(i - 5) * width + j - 3] +
					F[3] * (float)a[(i - 5) * width + j - 2] + F[4] * (float)a[(i - 5) * width + j - 1] + F[5] * (float)a[(i - 5) * width + j] +
					F[6] * (float)a[(i - 5) * width + j + 1] + F[7] * (float)a[(i - 5) * width + j + 2] + F[8] * (float)a[(i - 5) * width + j + 3] +
					F[9] * (float)a[(i - 5) * width + j + 4] + F[10] * (float)a[(i - 5) * width + j + 5] + F[11] * (float)a[(i - 4) * width + j - 5] +
					F[12] * (float)a[(i - 4) * width + j - 4] + F[13] * (float)a[(i - 4) * width + j - 3] + F[14] * (float)a[(i - 4) * width + j - 2] +
					F[15] * (float)a[(i - 4) * width + j - 1] + F[16] * (float)a[(i - 4) * width + j] + F[17] * (float)a[(i - 4) * width + j + 1] +
					F[18] * (float)a[(i - 4) * width + j + 2] + F[19] * (float)a[(i - 4) * width + j + 3] + F[20] * (float)a[(i - 4) * width + j + 4] +
					F[21] * (float)a[(i - 4) * width + j + 5] + F[22] * (float)a[(i - 3) * width + j - 5] + F[23] * (float)a[(i - 3) * width + j - 4] +
					F[24] * (float)a[(i - 3) * width + j - 3] + F[25] * (float)a[(i - 3) * width + j - 2] + F[26] * (float)a[(i - 3) * width + j - 1] +
					F[27] * (float)a[(i - 3) * width + j] + F[28] * (float)a[(i - 3) * width + j + 1] + F[29] * (float)a[(i - 3) * width + j + 2] +
					F[30] * (float)a[(i - 3) * width + j + 3] + F[31] * (float)a[(i - 3) * width + j + 4] + F[32] * (float)a[(i - 3) * width + j + 5] +
					F[33] * (float)a[(i - 2) * width + j - 5] + F[34] * (float)a[(i - 2) * width + j - 4] + F[35] * (float)a[(i - 2) * width + j - 3] +
					F[36] * (float)a[(i - 2) * width + j - 2] + F[37] * (float)a[(i - 2) * width + j - 1] + F[38] * (float)a[(i - 2) * width + j] +
					F[39] * (float)a[(i - 2) * width + j + 1] + F[40] * (float)a[(i - 2) * width + j + 2] + F[41] * (float)a[(i - 2) * width + j + 3] +
					F[42] * (float)a[(i - 2) * width + j + 4] + F[43] * (float)a[(i - 2) * width + j + 5] + F[44] * (float)a[(i - 1) * width + j - 5] +
					F[45] * (float)a[(i - 1) * width + j - 4] + F[46] * (float)a[(i - 1) * width + j - 3] + F[47] * (float)a[(i - 1) * width + j - 2] +
					F[48] * (float)a[(i - 1) * width + j - 1] + F[49] * (float)a[(i - 1) * width + j] + F[50] * (float)a[(i - 1) * width + j + 1] +
					F[51] * (float)a[(i - 1) * width + j + 2] + F[52] * (float)a[(i - 1) * width + j + 3] + F[53] * (float)a[(i - 1) * width + j + 4] +
					F[54] * (float)a[(i - 1) * width + j + 5] + F[55] * (float)(float)a[i * width + j - 5] + F[56] * (float)a[i * width + j - 4] +
					F[57] * (float)a[i * width + j - 3] + F[58] * (float)a[i * width + j - 2] + F[59] * (float)a[i * width + j - 1] +
					F[60] * (float)a[i * width + j] + F[61] * (float)a[i * width + j + 1] + F[62] * (float)a[i * width + j + 2] +
					F[63] * (float)a[i * width + j + 3] + F[64] * (float)a[i * width + j + 4] + F[65] * (float)a[i * width + j + 5] +
					F[66] * (float)a[(i + 1) * width + j - 5] + F[67] * (float)a[(i + 1) * width + j - 4] + F[68] * (float)a[(i + 1) * width + j - 3] +
					F[69] * (float)a[(i + 1) * width + j - 2] + F[70] * (float)a[(i + 1) * width + j - 1] + F[71] * (float)a[(i + 1) * width + j] +
					F[72] * (float)a[(i + 1) * width + j + 1] + F[73] * (float)a[(i + 1) * width + j + 2] + F[74] * (float)a[(i + 1) * width + j + 3] +
					F[75] * (float)a[(i + 1) * width + j + 4] + F[76] * (float)a[(i + 1) * width + j + 5] + F[77] * (float)a[(i + 2) * width + j - 5] +
					F[78] * (float)a[(i + 2) * width + j - 4] + F[79] * (float)a[(i + 2) * width + j - 3] + F[80] * (float)a[(i + 2) * width + j - 2] +
					F[81] * (float)a[(i + 2) * width + j - 1] + F[82] * (float)a[(i + 2) * width + j] + F[83] * (float)a[(i + 2) * width + j + 1] +
					F[84] * (float)a[(i + 2) * width + j + 2] + F[85] * (float)a[(i + 2) * width + j + 3] + F[86] * (float)a[(i + 2) * width + j + 4] +
					F[87] * (float)a[(i + 2) * width + j + 5] + F[88] * (float)a[(i + 3) * width + j - 5] + F[89] * (float)a[(i + 3) * width + j - 4] +
					F[90] * (float)a[(i + 3) * width + j - 3] + F[91] * (float)a[(i + 3) * width + j - 2] + F[92] * (float)a[(i + 3) * width + j - 1] +
					F[93] * (float)a[(i + 3) * width + j] + F[94] * (float)a[(i + 3) * width + j + 1] + F[95] * (float)a[(i + 3) * width + j + 2] +
					F[96] * (float)a[(i + 3) * width + j + 3] + F[97] * (float)a[(i + 3) * width + j + 4] + F[98] * (float)a[(i + 3) * width + j + 5] +
					F[99] * (float)a[(i + 4) * width + j - 5] + F[100] * (float)a[(i + 4) * width + j - 4] + F[101] * (float)a[(i + 4) * width + j - 3] +
					F[102] * (float)a[(i + 4) * width + j - 2] + F[103] * (float)a[(i + 4) * width + j - 1] + F[104] * (float)a[(i + 4) * width + j] +
					F[105] * (float)a[(i + 4) * width + j + 1] + F[106] * (float)a[(i + 4) * width + j + 2] + F[107] * (float)a[(i + 4) * width + j + 3] +
					F[108] * (float)a[(i + 4) * width + j + 4] + F[109] * (float)a[(i + 4) * width + j + 5] + F[110] * (float)a[(i + 5) * width + j - 5] +
					F[111] * (float)a[(i + 5) * width + j - 4] + F[112] * (float)a[(i + 5) * width + j - 3] + F[113] * (float)a[(i + 5) * width + j - 2] +
					F[114] * (float)a[(i + 5) * width + j - 1] + F[115] * (float)a[(i + 5) * width + j] + F[116] * (float)a[(i + 5) * width + j + 1] +
					F[117] * (float)a[(i + 5) * width + j + 2] + F[118] * (float)a[(i + 5) * width + j + 3] + F[119] * (float)a[(i + 5) * width + j + 4] +
					F[120] * (float)a[(i + 5) * width + j + 5];
				if (total > nnInputs[f]) {
					nnInputs[f] = total;
					filterMapMaxI[f] = i;
					filterMapMaxJ[f] = j;
				}
			}
		}
		break;
	}
}

// compute feature map for all convolutional filters and the 1 image pixel color array, used for K grayscale and L grayscale
void convolve1() {
	int prev = 0;
	for (f = 0; f < numFilters; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c1);
	}
}

// compute feature map for all convolutional filters and the 3 image pixel color arrays, used for RGB, CMY, HSV, and HSL
void convolve3() {
	for (f = 0; f < numFiltersPerColor; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c1);
	}
	for (f = numFiltersPerColor; f < numFiltersPerColor * 2; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c2);
	}
	for (f = numFiltersPerColor * 2; f < numFilters; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c3);
	}
}

// compute feature map for all convolutional filters and the 4 image pixel color arrays, used for RGBK and CMYK
void convolve4() {
	for (f = 0; f < numFiltersPerColor; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c1);
	}
	for (f = numFiltersPerColor; f < numFiltersPerColor * 2; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c2);
	}
	for (f = numFiltersPerColor * 2; f < numFiltersPerColor * 3; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c3);
	}
	for (f = numFiltersPerColor * 3; f < numFilters; f++) {
		nnInputs[f] = -999999999.0f;
		F = filter[f];
		convolveColor(c4);
	}
}

// executes either convolve1, convolve3, or convolve4 depending on the number of colors in the current color model
void convolve() {
	char nc = getNumColors();
	switch (nc) {
	case 1:
		convolve1();
		break;
	case 3:
		convolve3();
		break;
	case 4:
		convolve4();
		break;
	}
	// subtract average of previous feature presence values from each neural network input, update the average as part of training
	int totalLength = imageNumber + 1;
	if (imageNumber < numTraining) {
		for (int i = 0; i < numFilters; i++) {
			nnInputTotals[i] += nnInputs[i];
			nnInputs[i] -= nnInputTotals[i] / totalLength;
		}
	}
	else {
		totalLength = numTraining;
		for (int i = 0; i < numFilters; i++) {
			nnInputs[i] -= nnInputTotals[i] / totalLength;
		}
	}
}

// compute neural network hidden layer and outputs from inputs, weights, and biases; return greatest output value as the classification prediction as to the traffic sign type
char computeNN() {
	float max = -999999999.0f;
	char maxIndex = -1;
	for (int i = 0; i < numFilters; i++) {
		nnHidden[i] = nnBiases1[i];
		for (int j = 0; j < numFilters; j++) {
			nnHidden[i] += nnInputs[j] * nnWeights1[i][j];
		}
		if (nnHidden[i] < 0.0f) {
			nnHidden[i] = 0.0f;
		}
	}
	for (int i = 0; i < 14; i++) {
		nnOutputs[i] = nnBiases2[i];
		for (int j = 0; j < numFilters; j++) {
			nnOutputs[i] += nnHidden[j] * nnWeights2[i][j];
		}
		if (nnOutputs[i] > max) {
			max = nnOutputs[i];
			maxIndex = i;
		}
	}
	return maxIndex + 1;
}

// trains the CNN after a classification is done
void train(char prediction, char correct) {
	char pr = prediction - 1;
	char co = correct - 1;

	int colorsGreater = 0;

	for (int i = 0; i < numFilters; i++) {
		
		// selecting the pixel color array that applies to the convolutional filter being trained
		unsigned char* current = NULL;
		if (i < numFiltersPerColor) {
			current = c1;
		}
		else {
			if (i < numFiltersPerColor * 2) {
				current = c2;
			}
			else {
				if (i < numFiltersPerColor * 3) {
					current = c3;
				}
				else {
					current = c4;
				}
			}
		}
		
		// changing filter values
		float av = 0.0f;
		for (int j = 0; j < filterArea; j++) {
			av += 0.0001f * (float)current[(filterMapMaxI[i] + (j / filterSize) - padding) * width + filterMapMaxJ[i] + (j % filterSize) - padding] / (float)(imageNumber + 1);
		}
		for (int j = 0; j < filterArea; j++) {
			filter[i][j] += 0.0001f * (float)current[(filterMapMaxI[i] + (j / filterSize) - padding) * width + filterMapMaxJ[i] + (j % filterSize) - padding] / (float)(imageNumber + 1);
			filter[i][j] -= av / (float)filterArea;
		}
	}
	
	// finding the minimum, maximum, range, and normalizing the output values for training weights and biases
	float P[14] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
	float max = nnOutputs[0];
	float min = nnOutputs[0];
	for (int i = 1; i < 14; i++) {
		if (nnOutputs[i] > max) {
			max = nnOutputs[i];
		}
		if (nnOutputs[i] < min) {
			min = nnOutputs[i];
		}
	}
	float range = max - min;
	if (range < 10000.0f) {
		range = 10000.0f;
	}
	for (int i = 0; i < 14; i++) {
		P[i] = (nnOutputs[i] - min) / range;
	}
	
	float tr = 0.005f;

	// training the second half bias values
	for (int i = 0; i < 14; i++) {
		if (co == i) {
			nnBiases2[i] -= tr * ((200.0f * P[i] - 200.0f) / range);
		}
		else {
			nnBiases2[i] -= tr * (2.0f * P[i] / range);
		}
	}

	// training the second half weight values
	for (int i = 0; i < 14; i++) {
		for (int j = 0; j < numFilters; j++) {
			if (co == i) {
				nnWeights2[i][j] -= tr * (nnHidden[j] * (200.0f * P[i] - 200.0f) / range);
			}
			else {
				nnWeights2[i][j] -= tr * (nnHidden[j] * 2.0f * P[i] / range);
			}
		}
	}

	// values to change the biases and weights on the first half of the neural network by; these depend on the ideal change in output values
	float change[maxNumFilters] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };

	// training the first half bias values
	for (int i = 0; i < numFilters; i++) {
		for (int j = 0; j < 14; j++) {
			if (co == j) {
				change[i] += ((200.0f * P[j] - 200.0f) / range) * nnWeights2[j][i];
			}
			else {
				change[i] += (2.0f * P[j] / range) * nnWeights2[j][i];
			}
		}
		nnBiases1[i] -= tr * change[i];
	}

	// training the first half weight values
	for (int h = 0; h < numFilters; h++) {
		for (int i = 0; i < numFilters; i++) {
			nnWeights1[h][i] -= tr * change[h] * nnInputs[i];
		}
	}
}

// function used to test getRand
void testRand() {
	printf("100 random integers from 0 to 99:\n");
	for (int i = 0; i < 100; i++) {
		printf("%i ", randInt(0, 100));
	}
	printf("\n\n100 random decimals from -50 to 50:\n");
	for (int i = 0; i < 100; i++) {
		printf("%f ", randFloat(-50.0f, 100.0f));
	}
	printf("\n\n");
}

// function used to test getAddress
void testAddressConstructor() {
	for (int i = 1; i < 13; i++) {
		for (int j = 1; j <= 5; j++) {
			printf("%s\n", getAddress(i, j, 7, 234));
		}
	}
}

// function used to test whether the number of images of each sign type in each folder is as in imageCounts
void testNumImages() {

	FILE* f;

	int numChallengeLevels = 5;
	int n = 0;
	int challengeIndex = 0;

	int l = 0;

	char err = 0;

	// visual conditions
	for (int i = 0; i < 13; i++) {
		if (i == 0) {
			numChallengeLevels = 1;
		}
		else {
			numChallengeLevels = 5;
		}

		// challenge levels
		for (int j = 1; j <= numChallengeLevels; j++) {

			// sign types
			for (int k = 1; k < 15; k++) {

				l = imageCounts[k - 1]; 
				fopen_s(&f, getAddress(i, j, k, l), "r");
				if (f == NULL) {
					printf("File %s is NULL - %i %i %i %i\n", getAddress(i, j, k, l), i, j, k, l);
					err = 1;
					break;
				}
				else {
					fclose(f);
				}
			}
		}
	}
	if (!err) {
		printf("All folders contain the needed images.\n");
	}
}

// function used to test the image randomizer
void testRandomImages() {
	initializeImagesAll();
	for (int i = 0; i < numTotal; i++) {
		printf("Image %i: Condition %i, Challenge Level %i, Sign Type %i, Sign Number %i\n", i, imageConditions[i], imageChallenges[i], imageSigns[i], imageNumbers[i]);
	}
}

// function used to test reading an image file and printing the pixel color values
void testFileReading() {

	readFile("C:\\Train\\Darkening-1\\01_11_04_01_0052.bmp");

	printf("Inner Width: %i, Inner Height: %i, Line Length: %i\n\n", innerWidth, innerHeight, lineLength);

	printf("\n\nBGR from left to right, bottom to top:\n\n");

	for (int h = 0; h < innerHeight; h++) {
		for (int i = 0; i < lineLength; i++) {
			printf("%i ", file[h * lineLength + i + 54]);
		}
		printf("\n");
	}
}

// function used to test convert
void testConversions() {
	r[0] = 0; g[0] = 20; b[0] = 20;
	r[1] = 40; g[1] = 20; b[1] = 20;
	r[2] = 60; g[2] = 0; b[2] = 20;
	r[3] = 200; g[3] = 80; b[3] = 80;
	r[4] = 80; g[4] = 80; b[4] = 150;
	r[5] = 255; g[5] = 20; b[5] = 255;
	r[6] = 0; g[6] = 90; b[6] = 0;
	r[7] = 0; g[7] = 0; b[7] = 0;
	r[8] = 129; g[8] = 255; b[8] = 183;

	height = 3;
	width = 3;
	printf("Nine example colors converted to eight color models:\n\n");
	for (char i = 0; i < 8; i++) {
		currentColorModel = i;
		convert();
		for (int i = 0; i < 9; i++) {
			printf("R %i, G %i, B %i, C1 %i, C2 %i, C3 %i, C4 %i\n", r[i], g[i], b[i], c1[i], c2[i], c3[i], c4[i]);
		}
		printf("\n\n");
	}
}

// runs the entire experimental process: 5 filter sizes * 4 filter counts * 8 color models = 160 trials 
void runTest() {
	char prediction = -1;
	
	printf("Starting tests... (This will probably take a few minutes per test.)\n");
	printf("Test results will be displayed after each test concludes.\n\n");

	for (filterSize = 3; filterSize < 12; filterSize += 2) {

		filterArea = filterSize * filterSize;
		halfFilterArea = filterArea / 2;
		padding = filterSize / 2;
		doublePadding = padding * 2;

		for (numFilters = 24; numFilters < 100; numFilters += 24) {

			for (currentColorModel = 0; currentColorModel < 8; currentColorModel++) {
				imagesCorrect = 0;
				imagesClassified = 0;
				randomizeParameters();

				numFiltersPerColor = numFilters / getNumColors();
				
				memoryUsage = 4 * (numFilters * filterArea + numFilters * (numFilters + 14 + 3) + (2 * 14)) + (3 + getNumColors()) * maxImageSize + maxFileSize;

				start = (int)clock();

				// classify all images
				for (imageNumber = 0; imageNumber < numTotal; imageNumber++) {

					// get address of image, read all file content and store image pixel color data
					readFile(getAddress(imageConditions[imageNumber], imageChallenges[imageNumber], imageSigns[imageNumber], imageNumbers[imageNumber]));
					// convert image pixel color data to this trial's color model
					convert();
					// convolve the image pixels
					convolve();
					// use the neural network to compute the classification prediction
					prediction = computeNN();

					// if in the training phase, train the algorithm
					if (imageNumber < numTraining) {
						train(prediction, imageSigns[imageNumber]);
					}
					else {
						// otherwise, measure classification accuracy
						if (imageNumber == numTraining) {
							stop = (int)clock();
							timeTraining = stop - start;
							start = stop;
						}
						if (prediction == imageSigns[imageNumber]) {
							imagesCorrect++;
						}
						imagesClassified++;
					}
				}
				stop = (int)clock();

				// display testing results
				timeTesting = stop - start;
				timeTotal = timeTraining + timeTesting;
				printf("%i Filters of Size %ix%i, Color Model #%i: %i/%i (%f%%)\nTraining Duration: %ims\nTesting Duration: %ims\nTotal Duration: %ims (%fms per image)\nTotal Memory Usage: %i bytes\n\n",
				numFilters, filterSize, filterSize, currentColorModel + 1, imagesCorrect, imagesClassified, 100.0f * (float)imagesCorrect / (float)imagesClassified, timeTraining, timeTesting, timeTotal, ((float)timeTotal) / (float)numTotal, memoryUsage);
			}
		}
	}
	printf("All tests have finished.\n\n");
}

int main(void) {

	setup();

	//testRand();
	//testAddressConstructor();
	//testNumImages();
	//testRandomImages();
	//testFileReading();
	//testConversions();

	initializeImagesAll();
	//initializeImagesChallengeFree();
	//initializeImagesLowChallenge();

	runTest();

	return 0;
}
