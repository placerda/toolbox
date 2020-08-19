import pydicom
import nibabel as nb
import SimpleITK as sitk
import numpy as np
import cv2

CT_SIZE = 512

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

    # remove padding (don't know why some slices come with padding)
    if ds.Rows > CT_SIZE:
        row_padding = max(ds.Rows - 512, 0) // 2
        slice = slice[row_padding:-row_padding,:]
    if ds.Columns > CT_SIZE:
        column_padding = max(ds.Columns - 512, 0) // 2
        slice = slice[:, column_padding:-column_padding]

    jpeg = quantize_hu_rgb(slice, window_length, window_width)

    return jpeg

def nifti_to_jpeg(nifti_file, num_slices=1, window_length=-600, window_width=1500):
    """
    Reads NifTI file and returns middle slice in JPEG equivalent format
    :param nifti_file: NifTI file path
    :param num_slices: Number of slices to return (from the middle)
    :param window_length: HU window length, by default uses lung best window
    :param window_width: HU window width, by default uses lung best window
    :return: list of ndarrays of shape (n, n, 3)
    """
    # load nifti
    img = nb.load(nifti_file)
    data = img.get_fdata()
    # get middle slice
    mean = data.shape[2] // 2
    start = mean - (num_slices // 2)
    jpegs = []
    for i in range(start, start + num_slices):
        slice = data[:, :, i]
        slice = np.rot90(slice) # rotates nifti
        jpeg = quantize_hu_rgb(slice, window_length, window_width)
        jpegs.append(jpeg)

    return jpegs

def raw_to_jpeg(mhd_file, num_slices=1, window_length=-600, window_width=1500):
    """
    Reads mhd file and returns middle slice in JPEG equivalent format
    :param mhd_file: mhd file path
    :param window_length: HU window length, by default uses lung best window
    :param window_width: HU window width, by default uses lung best window
    :return: ndarray of shape (n, n, 3)
    """
    # load raw
    image = sitk.ReadImage(mhd_file)
    ct_scan = sitk.GetArrayFromImage(image)  # coordinates in z,y,x format

    # get middle slice
    mean = ct_scan.shape[0] // 2
    start = mean - (num_slices // 2)
    jpegs = []
    for i in range(start, start + num_slices):
        slice = ct_scan[i, :, :]
        jpeg = quantize_hu_rgb(slice, window_length, window_width)
        jpegs.append(jpeg)

    return jpeg