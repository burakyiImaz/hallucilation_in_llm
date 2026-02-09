import matplotlib.pyplot as plt


class BasePlotter:
    def __init__(self, style="default", figsize=(8, 5)):
        plt.style.use(style)
        self.figsize = figsize

    def _setup(self, title, xlabel=None, ylabel=None):
        plt.figure(figsize=self.figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    def show(self):
        plt.tight_layout()
        plt.show()
