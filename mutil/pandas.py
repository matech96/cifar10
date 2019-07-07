def _series_to_categorical(s):
    img_classes_int = s.astype('category').cat.codes.tolist()
    return img_classes_int
