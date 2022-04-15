import SimpleITK
import numpy as np
import monai_unet

class Unet_basline():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        pass

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        pass
        #return inputs

    def write_outputs(self, outputs):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        pass
    
    def predict(self, inputs):
        """
        Your algorithm goes here
        """        
        pass
        #return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        #inputs = self.load_inputs()
        #outputs = self.predict(inputs)
        #self.write_outputs(outputs)
        monai_unet.run_inference()

if __name__ == "__main__":
    Unet_basline().process()
