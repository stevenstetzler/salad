from ..serialize import read
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.afw.image as afwImage
import numpy as np
from scipy.signal import convolve2d, sepfir2d
from scipy.ndimage import gaussian_filter

def main():
    images = read("/epyc/projects/salad/search/tno_search/DEEP/20190403/A0c/detector_1/images.pkl")
    exposure = images[0].exposure
    maskedImage = exposure.maskedImage
    image = maskedImage.image
    variance = maskedImage.variance

    psf = exposure.getPsf()
    bbox = exposure.getBBox()
    detection = measAlg.SourceDetectionTask()
    result = detection.convolveImage(exposure.getMaskedImage(), psf)
    convolved = result.middle

    sigma = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
    kWidth = detection.calculateKernelSize(sigma)
    gaussFunc = afwMath.GaussianFunction1D(sigma)
    gaussKernel = afwMath.SeparableKernel(kWidth, kWidth, gaussFunc, gaussFunc)
    
    convolvedImage = image.Factory(bbox)
    afwMath.convolve(convolvedImage, image, gaussKernel, afwMath.ConvolutionControl())
    convolvedVariance = variance.Factory(bbox)
    afwMath.convolve(convolvedVariance, variance, gaussKernel, afwMath.ConvolutionControl())
    
    a = np.exp(-(np.arange(-10, 11) / sigma)**2).astype(np.float32)
    a /= a.sum()
    # print("a", a)
    # scipy_convolvedImage = convolve2d(image.array, psf.computeImage(bbox.getCenter()).array, mode='same', boundary='fill', fillvalue=0)
    # scipy_convolvedImage = gaussian_filter(image.array, sigma, mode='constant', cval=0)[41:2048-41, 41:4096-41]
    scipy_convolvedImage = sepfir2d(image.array, a, a)
    # scipy_convolvedVariance = convolve2d(variance.array, psf.computeImage(bbox.getCenter()).array**2, mode='same', boundary='fill', fillvalue=0)
    # scipy_convolvedVariance = gaussian_filter(variance.array, sigma, mode='constant', cval=0)[41:2048-41, 41:4096-41]
    scipy_convolvedVariance = sepfir2d(variance.array, a**2, a**2) / 2 # not sure why this is / 2

    print("image")
    print("image", image)
    print("convolved.image", convolved.image)
    print("convolvedImage", convolvedImage.Factory(convolvedImage, convolved.getBBox(), afwImage.PARENT, False))
    print("scipy_convolvedImage", scipy_convolvedImage)

    print("variance")
    print("variance", variance)
    print("convolved.variance", convolved.variance)
    print("convolvedVariance", convolvedVariance.Factory(convolvedVariance, convolved.getBBox(), afwImage.PARENT, False))
    print("scipy_convolvedVariance", scipy_convolvedVariance)




if __name__ == "__main__":
    main()
