

NUMBER_OF_OUTPUT_CLASSES = 2

class OutputClass:
    def __init__(self, label, value):
        self.label = label
        self.value = value
        return


#  length of array is NUMBER_OF_OUTPUT_CLASSES
def get_new_output_class_array():
    return [0] * NUMBER_OF_OUTPUT_CLASSES
