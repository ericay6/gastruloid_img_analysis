
import numpy as np
import matplotlib.pyplot as plt

# from src.analysis.medialaxis import expand_skel, get_length
import src.image.imUtilsJ as imuj

import javabridge as jv
VM_STARTED = False
VM_KILLED = False
import bioformats

def init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    rootLoggerName = jv.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jv.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jv.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jv.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)

def start_jvm(max_heap_size='4G'):
    """
    Start the Java Virtual Machine, enabling BioFormats IO.
    Optional: Specify the path to the bioformats_package.jar to your needs by calling.
    set_bfpath before staring to read the image data
    Parameters
    ----------
    max_heap_size : string, optional
    The maximum memory usage by the virtual machine. Valid strings
    include '256M', '64k', and '2G'. Expect to need a lot.
    """

    jv.start_vm(class_path=bioformats.JARS, max_heap_size=max_heap_size)
    init_logger()
    VM_STARTED = True

def kill_jvm():
    """
    Kill the JVM. Once killed, it cannot be restarted.
    See the python-javabridge documentation for more information.
    """
    jv.kill_vm()
    VM_KILLED = True


def get_metadata(filename):
    '''
    from a czi or vsi, I want to get multiple informations :
    -number of pixels
    -physical pixel size
    '''

    if not VM_STARTED:
        start_jvm()

    md              = bioformats.get_omexml_metadata( filename )
    metadata = {
        #         "fullmetadata" : md,
        "Nx"        : bioformats.OMEXML(md).image().Pixels.SizeX,
        "Ny"        : bioformats.OMEXML(md).image().Pixels.SizeY,
        "Nz"        : bioformats.OMEXML(md).image().Pixels.SizeZ,
        "Nt"        : bioformats.OMEXML(md).image().Pixels.SizeT,
        "Nch"       : bioformats.OMEXML(md).image().Pixels.SizeC,
        "dtype_str" : bioformats.OMEXML(md).image().Pixels.PixelType,
        "dx"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeX,
        "dxUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeXUnit,
        "dy"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeY,
        "dyUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeYUnit,
        "dz"        : bioformats.OMEXML(md).image().Pixels.PhysicalSizeZ,
        "dzUnit"    : bioformats.OMEXML(md).image().Pixels.PhysicalSizeZUnit}
    try:
        metadata["Magn"] = bioformats.OMEXML(md).instrument().Objective.NominalMagnification
    except:
        print('no magn found')
    try:
        metadata["Nseries"]= bioformats.ImageReader(filename).rdr.getSeriesCount()
    except:
        metadata['Nseries']=1
        print('#series not found')

    return metadata


def binarize(image):

    imeq = (image - np.min(image) ) / ( np.max(image) - np.min(image))

    from skimage import filters
    from skimage import morphology
    from skimage import exposure

    image = exposure.adjust_gamma(imeq,0.75)
    image = filters.sobel(image)
    img = filters.gaussian(image, sigma=5)

    val = filters.threshold_otsu(img)


    binary = np.zeros_like(img, dtype=int)
    binary[img > val] = 1

    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary

    filled = morphology.reconstruction(seed, mask, method='erosion')

    return np.int0(filled)

def binarize_BF(image):

    from skimage import exposure, util, filters
    from skimage.morphology import binary_opening, reconstruction
    from skimage.morphology.selem import disk

    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    Iinv = util.invert(img_rescale)

    val = filters.threshold_otsu(Iinv)
    binary = np.zeros_like(Iinv, dtype=int)
    binary[Iinv > val] = 1

    img = binary_opening(binary, disk(5) )

    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary
    filled = reconstruction(seed, mask, method='erosion')

    return np.int0(filled)

def binarize_percenthist(image, thres = 0.05, selem_rad=30):


    from skimage.morphology import binary_erosion, binary_dilation, erosion
    from skimage.morphology.selem import disk
    from skimage.exposure import cumulative_distribution
    from scipy.ndimage import binary_fill_holes

    # image_ero = erosion(image,selem=disk(3))

    H, bins= cumulative_distribution(image, nbins=256)
    val = bins[np.argwhere(H<thres)[-1]+1][0]
    bin = np.zeros_like(image)
    bin[image < val] =1

    image_dil = binary_dilation(bin, selem = disk(selem_rad))
    image_filled = binary_fill_holes(image_dil)
    image_fin = binary_erosion(image_filled, selem=disk(selem_rad))

    return image_fin

def clean_binarized(filled):

    from skimage import morphology
    from skimage.measure import label
    from skimage.segmentation import clear_border

    labels = label(filled)
    eroded = morphology.binary_erosion(labels, footprint=morphology.disk(15))
    cleared = morphology.remove_small_objects(eroded, 10000)
    dilated = morphology.binary_dilation(cleared, footprint=morphology.disk(15))
    img=np.zeros_like(filled)
    img[dilated[:]] = 1
    img= clear_border(img)


    return img

def get_background(image, mask):
    ''' Function that calculate the mean values of the image outside of mask as an estimate of the background
    This value is used for instance to fill the image after rotating them when virtually aligning gastruloids'''

    mask = imuj.snake_to_bw(mask, image.shape)
    mask = np.invert(mask)
    background = np.mean(image[np.nonzero(mask)])

    return background



def get_contour(binary, image, returnImage = False, alpha = 0.1, beta=20, w_line=0, w_edge=0):

    from skimage.segmentation import active_contour

    snake_init=imuj.bw_to_snake(binary)
    #TODO : change active_contour or optimize its parameters
    snake = active_contour(image,snake_init, alpha=alpha, beta=beta, w_line=w_line, w_edge=w_edge )
    print('active contour done')

    if returnImage == True:
        bw_snake = imuj.snake_to_bw(snake,binary.shape)
    else:
        bw_snake = []

    return snake, bw_snake


# def get_equivalent_radius(label):

def maxprojection(czifile, theS=0, theC=0, theT=0):
    ''' Calculate the max projection of a subset of images in a czifile'''

    if VM_KILLED:
        print('error, JVM was killed before')
    elif not VM_STARTED:
        start_jvm()

    reader = bioformats.ImageReader(czifile)
    metadata = imuj.GetMetadata(czifile)
    Nz= metadata["Nz"]

    I = reader.read(rescale=False, z=0, c=theC, series=theS, t=theT)

    maxi = np.copy(I)
    indz = np.zeros_like(I)

    for theZ in range(1,Nz):
        I = reader.read(rescale=False, z=theZ, c=theC, series=theS)
        inds = I > maxi
        indz[inds] = theZ
        maxi[inds] = I[inds]  # update the maximum value at each pixel

    return maxi, indz


