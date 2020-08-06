import pydicom
import nibabel as nb
import numpy as np
import cv2

def quantize_hu_rgb(hu_data, window_length=-600, window_width=1500):
    """
    Quantize HU in 256
    :param hu_data: ndarray of shape (m, n, 1)
    :param window_length: DICOM's window length, by default uses lung best window
    :param window_width: DICOM's window width, by default uses lung best window
    :return: ndarray of shape (m, n, 3) type uint8
    """

    # quantization parameters
    WINDOW_LENGHT=window_length
    WINDOW_WIDTH=window_width
    MIN_HU=-1024
    MAX_HU=1024

    # 256 quantization
    min_value = WINDOW_LENGHT - (WINDOW_WIDTH // 2)
    if min_value < MIN_HU: min_value = MIN_HU
    max_value = WINDOW_LENGHT + (WINDOW_WIDTH // 2)
    if max_value > MAX_HU: max_value = MAX_HU
    quantized = hu_data.copy()
    quantized = np.clip(quantized, min_value, max_value)
    for cell in np.nditer(quantized, op_flags=['readwrite']):
        cell[...] = ((cell - min_value) * 255) // (max_value - min_value)
    gray = quantized.astype('uint8')
    image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return image_rgb

def dicom_to_jpeg(dicom_file, window_length=-600, window_width=1500):
    """
    Reads DICOM file and returns it in JPEG equivalent format
    :param dicom_file: DICOM file path
    :param window_length: HU window length
    :param window_width: HU window width
    :return: ndarray of shape (n, n, 3)
    """
    # load dicom
    ds = pydicom.dcmread(dicom_file)
    b = ds.RescaleIntercept
    m = ds.RescaleSlope
    slice = m * ds.pixel_array + b

    jpeg = quantize_hu_rgb(slice, window_length, window_width)

    return jpeg

def nifti_to_jpeg(nifti_file, window_length=-600, window_width=1500):
    """
    Reads NifTI file and returns middle slice in JPEG equivalent format
    :param nifti_file: NifTI file path
    :param window_length: HU window length, by default uses lung best window
    :param window_width: HU window width, by default uses lung best window
    :return: ndarray of shape (n, n, 3)
    """
    # load nifti
    img = nb.load(nifti_file)
    data = img.get_fdata()

    # get middle slice
    mean = data.shape[2] // 2
    slice = data[:,:,mean]

    # rotates nifti
    slice = np.rot90(slice)
    jpeg = quantize_hu_rgb(slice, window_length, window_width)

    return jpeg