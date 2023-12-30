import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt


def sobel_kernel() -> np.ndarray:
    """
    Create discrete Sobel kernel
    """
    return (np.array([1, 2, 1]), np.array([-1, 0, 1]))


def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Create discrete Gaussian kernel, clip to ±3σ
    :param sigma: sigma is the std-deviation and refers to spread of gaussian
    """
    width = np.ceil(3 * sigma)
    support = np.arange(-width, width + 1)  # off by one for upper bound
    gauss_kernel = np.exp(-(support**2) / (2.0 * sigma**2)) / (
        sigma * np.sqrt(2 * np.pi)
    )
    return gauss_kernel


def symmetric_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with kernel, clip to ±3σ
    :param img: input image
    :param kernel: kernel to convolve with
    """
    tmp = np.zeros_like(img)
    gauss = np.zeros_like(img)
    for i in range(img.shape[0]):
        tmp[i, :] = np.convolve(img[i, :], kernel, mode="same")
    for j in range(img.shape[1]):
        gauss[:, j] = np.convolve(tmp[:, j], kernel, mode="same")
    return gauss


def convolve(img: np.ndarray, kernel_x: np.ndarray, kernel_y) -> np.ndarray:
    """
    Convolve image with kernel
    :param img: input image
    :param kernel: kernel to convolve with
    """
    tmp = np.zeros_like(img)
    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        tmp[i, :] = np.convolve(img[i, :], kernel_x, mode="same")
    for j in range(img.shape[1]):
        res[:, j] = np.convolve(tmp[:, j], kernel_y, mode="same")
    return res


def non_maximum_suppression(magnitude: np.ndarray, size: int = 2) -> np.ndarray:
    """
    Non maximum suppression
    :param magnitude: magnitude of eigenvalues
    """
    nms = np.copy(magnitude)
    for i in range(magnitude.shape[0] - 1):
        for j in range(magnitude.shape[1] - 1):
            try:
                for m in range(i - size, i + size + 1):
                    for n in range(j - size, j + size + 1):
                        if magnitude[i, j] < magnitude[m, n]:
                            nms[i, j] = 0

            except IndexError:
                pass
    nms[nms > 0] = 1
    return nms


def HarrisCorner(
    img: np.ndarray, sigma: float, k: float, threshold: float
) -> np.ndarray:
    """
    Harris Corner Detection
    :param img: input image
    :param sigma: sigma is the std-deviation and refers to spread of gaussian
    :param k: Harris constant
    :param threshold: threshold for corner detection
    """
    sob_x, sob_y = sobel_kernel()
    dx = convolve(img, sob_x, sob_y)
    dy = convolve(img, sob_y, sob_x)

    # Gaussian Filter
    A = symmetric_convolve(dx**2, gaussian_kernel(sigma))
    B = symmetric_convolve(dy**2, gaussian_kernel(sigma))
    C = symmetric_convolve(dx * dy, gaussian_kernel(sigma))

    response = (A * B - C**2) - k * (A + B) ** 2
    response[response < threshold] = 0
    return response


def main():
    image = Image.open("Bikesgray.jpg")
    img = np.array(image.convert("L")) / 255.0
    strong_corners = HarrisCorner(img, 1, 0.1, 0.8)
    nms = non_maximum_suppression(strong_corners)

    plt.imshow(image)
    dots_x, dots_y = np.where(nms.T > 0)
    plt.scatter(dots_x, dots_y, color="red", s=2)  # Plot green dots
    plt.show()


if __name__ == "__main__":
    main()
