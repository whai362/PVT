from .image_list import ImageList


class ImageNet(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path):
        super(ImageNet, self).__init__(
            root, list_file, memcached, mclient_path)
