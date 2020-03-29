CLASS_LIST = ['bottle',
              'cup',
              'knife',
              'bowl',
              'wine glass',
              'fork',
              'spoon',
              'banana',
              'apple',
              'orange',
              'cake',
              'potted plant',
              'mouse',
              'keyboard',
              'laptop',
              'cell phone',
              'book',
              'clock',
              'chair',
              'table',
              'couch',
              'bed',
              'toilet',
              'tv',
              'microwave',
              'toaster',
              'refrigerator',
              'oven',
              'sink',
              'person',
              'background'
              ]

CLASS_IDS = {class_name: idx for idx, class_name in enumerate(CLASS_LIST)}

# Some helper synonyms, to handle cases where multiple words mean the same class
# This list is used when loading the ground truth to map it to the list above,
# you could add what you want
SYNONYMS = {
    'television': 'tv',
    'tvmonitor': 'tv',
    'tv monitor': 'tv',
    'computer monitor': 'tv',
    'coffee table': 'table',
    'dining table': 'table',
    'kitchen table': 'table',
    'desk': 'table',
    'stool': 'chair',
    'sofa': 'couch',
    'diningtable': 'dining table',
    'pottedplant': 'potted plant',
    'plant': 'potted plant',
    'cellphone': 'cell phone',
    'mobile phone': 'cell phone',
    'mobilephone': 'cell phone',
    'wineglass': 'wine glass',

    # background classes
    'none': 'background',
    'bg': 'background',
    '__background__': 'background'
}

def get_class_id(class_name):
    """
    Given a class string, find the id of that class
    This handles synonym lookup as well
    :param class_name: the name of the class being looked up (can be synonym from SYNONYMS)
    :return: an integer of the class' ID in the CLASS_LIST
    """
    class_name = class_name.lower()
    if class_name in CLASS_IDS:
        return CLASS_IDS[class_name]
    elif class_name in SYNONYMS:
        return CLASS_IDS[SYNONYMS[class_name]]
    return None

def get_nearest_class(potential_class_name):
    """
    Given a string that might be a class name,
    return a string that is definitely a class name.
    Again, uses synonyms to map to known class names
    :param potential_class_name: the queried class name
    :return: the actual class name as a string
    """
    potential_class_name = potential_class_name.lower()
    if potential_class_name in CLASS_IDS:
        return potential_class_name
    elif potential_class_name in SYNONYMS:
        return SYNONYMS[potential_class_name]
    return None
