import imutils

def sliding_window(image, step, window_size):
    # print(image)
    # print(step)
    # print(window_size)
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            # Trả về 1 generator image (list)
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def image_pyramid(image, scale=1.5, min_Size=(244, 244)):
    # Trả về 1 generator image (list)
    yield image

    # Lặp qua tất cả tấm ảnh
    while True:
        # Tính toán kích thước của ảnh tiếp theo trong kim tự tháp (pyramid)
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # Nếu ảnh đã được resized mà nhỏ hơn kích thước tối thiểu,
        # thì dừng cấu trúc kim tự tháp (pyramid)
        if image.shape[0] < min_Size[1] or image.shape[1] < min_Size[0]:
            break

        # Trả về danh sách ảnh trong kim tự tháp (pyramid)
        yield image

