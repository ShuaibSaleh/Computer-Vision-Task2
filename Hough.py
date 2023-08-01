# import cv2
from math import sqrt, atan2, pi , cos, sin
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import ( hough_line_peaks)
from PIL import Image, ImageDraw
from collections import defaultdict
import cv2

def hough_Accumulator_thetas_dist(image):
    Ny = image.shape[0]
    Nx = image.shape[1]
    Max_distance = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.linspace(-Max_distance, Max_distance, 2*Max_distance)
    accumulator = np.zeros((2 * Max_distance, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            if image[y,x] > 0:
              for k in range(len(thetas)):           
                    y_intersection = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(y_intersection) + Max_distance,k] += 1
            else:
                continue
    return accumulator, thetas, rhos
    

def save_hough_transform(accumulator ,thetas , rhos):
    plt.imshow(np.log(1 + accumulator),extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]],cmap='gray', aspect=1/1.5)
    plt.title('Hough transform')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("Hough_transform.jpg")




def save_hough_Line(image ,accumulator ,thetas ,rhos):
    angle_list=[]
    plt.imshow(image, cmap='gray')

    origin = np.array((0, image.shape[1]))

    for _, angle, dist in zip(*hough_line_peaks(accumulator, thetas, rhos)):
        angle_list.append(angle) 
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        plt.plot(origin, (y0, y1), '-r')
        plt.xlim(origin)
        plt.ylim((image.shape[0], 0))
        plt.axis('off')
        # plt.title('Detected lines')
        plt.tight_layout()
        # plt.savefig(f'./static/images/output/hough_line.jpg')
        plt.savefig(f'./static/images/output/hough_line.jpg' , bbox_inches='tight' ,  pad_inches=0)
    # Save output image
    path_output = f'./static/images/output/hough_line.jpg'
    resizeimage=cv2.imread(path_output)
    resizeimage=cv2.resize(resizeimage , (420 ,330) )
    cv2.imwrite(path_output, resizeimage)
    return path_output
        
def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


def circlehough(path):
# Load image:
    input_image = Image.open(path)

    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    # Find circles
    rmin = 24
    rmax =28
    steps = 100
    threshold = 0.4

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []

    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(v / steps, x, y, r)
            circles.append((x, y, r))

    for x, y, r in circles:
        draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

    # Save output image
    output_image.save(f'./static/images/output/result.png')
    path_output = f'./static/images/output/result.png'
    return path_output



# Note this function comlixty is to high O(n^4) so it does not work efficiently
def hough_ellipse(image, threshold=200):
    edges = cv2.Canny(image, 5, 200)
    height, width = edges.shape
    accumulator = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            if edges[y][x] == 255:
                for a in range(1, width//2):
                    b = int(a * np.sqrt((x - width/2)*2 + (y - height/2)*2) / max(width/2, height/2))
                    if b > 0:
                        for theta in range(0, 360):
                            theta_rad = np.deg2rad(theta)
                            x0 = int(x - a * np.cos(theta_rad))
                            y0 = int(y + b * np.sin(theta_rad))
                            if x0 >= 0 and x0 < width and y0 >= 0 and y0 < height:
                                accumulator[y0][x0] += 1
    
    ellipses = []
    for y in range(height):
        for x in range(width):
            if accumulator[y][x] > threshold:
                ellipses.append((x,y))
    
    return ellipses


