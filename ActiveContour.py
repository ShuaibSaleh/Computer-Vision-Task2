from inspect import ClosureVars
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
import cv2


class contour():

    def __init__(self, image, alpha, beta, gamma, num_points):

        self.image = np.array(image)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.snake_points = None
        self.length = 0
        self.num_points = num_points
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        r1 = (self.width//2)-math.ceil(0.15*self.width)
        r2 = (self.height//2)-math.ceil(0.15*self.height)
        radians = np.linspace(0, 2 * np.pi, self.num_points)
        self.snake_points = np.array([
            [math.ceil((self.image.shape[1]//2 + r1 * np.cos(radians))[i]),
             math.ceil((self.image.shape[0]//2 + r2 * np.sin(radians))[i])]
            for i in range(self.num_points)])

        self.length = self.contour_len()

    def contour_len(self):

        l = [self.getDistance(self.snake_points[i], self.snake_points[i+1])
             for i in range(self.num_points-1)]
        l.append(self.getDistance(self.snake_points[0], self.snake_points[-1]))

        return np.sum(l)

    def contour_area(self):

        x = []
        y = []
        for i in range(self.num_points):
            x.append(self.snake_points[i][0])
            y.append(self.snake_points[i][1])
        area = 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))
        return area

    def getDistance(self, first_point, second_point):
        return np.sqrt(np.sum((first_point-second_point) ** 2))

    def getContinuityEnergy(self, currPoint, prevPoint):
        # get average distance between snake points = snakelen/ # points
        avgDistance = self.length / self.num_points
        # get distance between the prev and the current
        dist = self.getDistance(prevPoint, currPoint)
        continuityEnergy = (abs(dist - avgDistance))**2
        return continuityEnergy

    def grad_energy(self, curr):
        gx = cv2.Sobel(self.image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(self.image, cv2.CV_32F, 0, 1)
        # mag, ang = cv2.cartToPolar(gx, gy)
        grad = -((gx[curr[1]][curr[0]])**2+(gy[curr[1]][curr[0]])**2)
        return grad

    def curv_energy(self, prevPoint, currPoint, nextPoint):
        curv_x = prevPoint[0] - 2*currPoint[0] + nextPoint[0]
        curv_y = prevPoint[1] - 2*currPoint[1] + nextPoint[1]
        curvature_energy = (curv_x**2 + curv_y**2)

        return curvature_energy

    def getTotalEnergy(self, curv, cont, grad):
        total = self.alpha*cont + self.beta*curv + self.gamma * grad
        return total

    def GenerateKernel(self, size):
        win = []
        nums = np.arange(-(size//2), (size//2)+1)
        for element in product(nums, repeat=2):
            win.append(element)
        return (win)

    def fit_snake(self, kernel_size=3):

        points_cpy = np.copy(self.snake_points)
        window = self.GenerateKernel(kernel_size)
        for p in range(self.num_points):
            Energy = []
            curr = [0, 0]
            prev = points_cpy[-1] if p == 0 else points_cpy[p-1]
            next = points_cpy[0] if p == self.num_points-1 else points_cpy[p+1]

            for win in window:
                curr[0] = points_cpy[p][0] + win[0] if points_cpy[p][0] + \
                    win[0] < self.image.shape[1] else self.image.shape[1] - 2
                curr[1] = points_cpy[p][1] + win[1] if points_cpy[p][1] + \
                    win[1] < self.image.shape[0] else self.image.shape[0] - 2
                cont = self.getContinuityEnergy(curr, prev)
                curv = self.curv_energy(prev, curr, next)
                grad = self.grad_energy(curr)
                Energy.append(self.getTotalEnergy(curv, cont, grad))

            minEnergy = np.argmin(Energy)
            idx_x = curr[0]+window[minEnergy][0]
            idx_y = curr[1]+window[minEnergy][1]
            newPoint = [idx_x, idx_y]
            self.snake_points[p] = np.array(newPoint)

    def draw_contour(self):

        for i in range(self.num_points-1):
            img = cv2.line(
                self.image, self.snake_points[i],  self.snake_points[i+1], (0, 0, 0), thickness=4)
        img = cv2.line(
            self.image, self.snake_points[0],  self.snake_points[-1], (0, 0, 0), thickness=4)
        plt.imshow(img, cmap="gray")
        # plt.show()
        img = './static/images/output/output.jpg'
        plt.savefig(img)


# --------------------------------another code for active contour--------------------------------------
# For more precise active contour


Image = cv2.imread('./static/images/input/apple.jpg', 1)
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
plt.imshow(image, cmap='gray')
img = np.array(image, dtype=np.float64)

IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
IniLSF[60:120, 60:120] = -1
IniLSF = -IniLSF

Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(Image)
plt.xticks([])
plt.yticks([])
plt.contour(IniLSF, [0], color='b', linewidth=2)
plt.draw()
plt.show(block=False)


def mat_math(intput, str):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output


def CV(LSF, img, mu, nu, epison, step):

    Drc = (epison / math.pi) / (epison*epison + LSF*LSF)
    Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/epison, "atan"))
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix*Ix+Iy*Iy, "sqrt")
    Nx = Ix / (s+0.000001)
    Ny = Iy / (s+0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    Length = nu*Drc*cur

    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu*(Lap - cur)

    s1 = Hea*img
    s2 = (1-Hea)*img
    s3 = 1-Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2))

    LSF = LSF + step*(Length + Penalty + CVterm)
    #plt.imshow(s, cmap ='gray'),plt.show()
    return LSF


def start():
    mu = 1
    nu = 0.003 * 255 * 255
    num = 20
    epison = 1
    step = 0.1
    LSF = IniLSF
    for i in range(1, num):
        LSF = CV(LSF, img, mu, nu, epison, step)
        if i % 1 == 0:
            plt.imshow(Image), plt.xticks([]), plt.yticks([])
            plt.contour(LSF, [0], colors='b', linewidth=2)
            plt.draw(), plt.show(block=False), plt.pause(0.01)
