import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name

classes = label_to_name

print(classes(0))