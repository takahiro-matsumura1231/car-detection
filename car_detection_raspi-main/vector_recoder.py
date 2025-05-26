import os

import numpy as np

class VectorRecorder:
    def __init__(self):
        self.recorded_vector = None

        self.save_vector_dir = "./saved_vector"
        self.save_vector_name = "vector.npz"
        os.makedirs(self.save_vector_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.save_vector_dir, self.save_vector_name)):
            self.recorded_vector = np.load(os.path.join(self.save_vector_dir, self.save_vector_name))["recorded_vector"]
    
    def save(self, vector):
        self.recorded_vector = vector
        np.savez_compressed(os.path.join(self.save_vector_dir, self.save_vector_name), recorded_vector=self.recorded_vector)

    def reset(self):
        self.recorded_vector = None
        if os.path.exists(os.path.join(self.save_vector_dir, self.save_vector_name)):
            os.remove(os.path.join(self.save_vector_dir, self.save_vector_name))

    def calc_distance(self, vector):
        if self.recorded_vector is None:
            return float('inf')
        
        return ((vector - self.recorded_vector) ** 2).mean()
    
    def is_same_vector(self, vector, boundary = 10.0):
        return self.calc_distance(vector) < boundary
    
    def is_saved_vector(self):
        return self.recorded_vector is not None