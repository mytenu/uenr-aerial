# Fake cv2 module to bypass OpenCV dependency in cloud

def imread(*args, **kwargs):
    return None

def imwrite(*args, **kwargs):
    return None

class dnn:
    pass
