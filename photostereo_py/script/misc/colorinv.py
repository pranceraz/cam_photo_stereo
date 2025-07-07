import cv2

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def invert_color(image_path):
    # Convert BGR to RGB
    image = cv2.imread(image_path)
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


if __name__ == "__main__":
    input_img = "combined_color_albedo.png"
    
    inverted_img = invert_color(input_img)
    
    resize = ResizeWithAspectRatio(inverted_img, width=1280)
    
    cv2.imshow("Inverted Image", resize)
    cv2.waitKey(0)
    