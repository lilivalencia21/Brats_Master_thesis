

def extract_tumor(segmentation):
    img = segmentation

    img[img >=1] = 1
    tumor = r
