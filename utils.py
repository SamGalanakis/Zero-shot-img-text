
class IncrementalAverage:
    def __init__(self):
        self.value = 0
        self.counter = 0

    def update(self,x):
        self.counter+=1
        self.value =self.value +  (x-self.value)/self.counter