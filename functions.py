import cv2

target_size = (100, 100)  # Desired output size (width, height)

def resize_pad(image):
    h, w = image.shape[:2]
    
    # Resize while preserving aspect ratio
    if h > w:
        img_resize = cv2.resize(image, (target_size[0], int(h * target_size[0] / w)), interpolation=cv2.INTER_AREA)
    else:
        img_resize = cv2.resize(image, (int(w * target_size[1] / h), target_size[1]), interpolation=cv2.INTER_AREA)
        
    # Pad to reach target size
    delta_h = target_size[1] - img_resize.shape[0]
    delta_w = target_size[0] - img_resize.shape[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img_padded = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
    
    return img_padded

# Define feature extraction function (using pixel intensities here)
def extract_features(image):
  # Flatten the image into a 1D vector
  return image.flatten()