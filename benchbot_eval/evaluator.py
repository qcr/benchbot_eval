class Evaluator:

    def __init__(self, filename):
        self.filename = filename

    def evaluate(self):
        print("Running evaluation on: %s" % self.filename)
        print("Evaluation complete.")
