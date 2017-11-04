class WordInVocab:

    def __init__(self, class_counts):
        self.class_counts = class_counts
        self.scores = []

    def set_scores(self, scores):
        self.scores = scores

    def get_scores(self):
        return self.scores

    def get_class_counts(self):
        return self.class_counts

    def set_class_counts(self, class_counts):
        self.class_counts = class_counts
