import cv2
import numpy as np
import sys
import colorProcess as cp

if (len(sys.argv) != 7):
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv) - 1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if (w1 < 0 or h1 < 0 or w2 <= w1 or h2 <= h1 or w2 > 1 or h2 > 1):
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if (inputImage is None):
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape  # bands == 3
W1 = round(w1 * (cols - 1))
H1 = round(h1 * (rows - 1))
W2 = round(w2 * (cols - 1))
H2 = round(h2 * (rows - 1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)
out = np.copy(inputImage)

luv = inputImage.astype(np.float64)
for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        b, g, r = tmp[i, j]
        luv[i, j] = cp.BGR2LUV(b, g, r)

# set L range to 0-100

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        L, u, v = luv[i][j]
        if L > 100:
            L = 100
        if L < 0:
            L = 0
        luv[i][j] = L, u, v

k = 101
fi = np.zeros((101), dtype=np.int)
hi = np.zeros((101), dtype=np.int)
newL = np.zeros((101), dtype=np.float)
for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        hi[int(luv[i][j][0])] += 1
fi[0] = hi[0]
for i in range(1, 101):
    fi[i] = fi[i - 1] + hi[i]

newL[0] = (fi[0] * 1.0 / 2) * (101 * 1.0 / fi[100])
for i in range(1, 101):
    newL[i] = ((fi[i] + fi[i - 1]) * 1.0 / 2) * (101 * 1.0 / fi[100])
    newL[i] = int(newL[i])

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        L, u, v = luv[i][j]
        L = newL[int(L)]
        luv[i][j] = L, u, v

# back to rbg
for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        L, u, v = luv[i][j]
        out[i][j] = cp.LUV2BGR(L, u, v)

# end of example of going over window

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows):
    for j in range(0, cols):
        b, g, r = out[i, j]
        outputImage[i, j] = b, g, r
cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
