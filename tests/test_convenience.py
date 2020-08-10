import unittest
import toolbox
from matplotlib import pyplot as plt

class TestToolbox(unittest.TestCase):
    def test_dicom_to_jpeg(self):
        filename = './tests/testdata/dummy.dcm'
        result = toolbox.dicom_to_jpeg(filename)
        plt.title('test_dicom_to_jpeg')
        plt.imshow(result)
        plt.show()
        self.assertEqual(len(result.shape), 3)

    def test_nifti_to_jpeg(self):
        filename = './tests/testdata/dummy.nii.gz'
        result = toolbox.nifti_to_jpeg(filename)
        plt.title('test_nifti_to_jpeg')
        plt.imshow(result)
        plt.show()
        self.assertEqual(len(result.shape), 3)

    def test_raw_to_jpeg(self):
        filename = './tests/testdata/dummy.mhd'
        result = toolbox.raw_to_jpeg(filename)
        plt.title('test_raw_to_jpeg')
        plt.imshow(result)
        plt.show()
        self.assertEqual(len(result.shape), 3)

if __name__ == '__main__':
    unittest.main()