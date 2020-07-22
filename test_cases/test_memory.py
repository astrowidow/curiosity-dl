if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from memory_profiler import memory_usage
from kdnn import Variable
from kdnn import no_grad
from kdnn.functions import square


class MemoryProf(unittest.TestCase):
    @staticmethod
    def cause_memory_leak():
        for i in range(100):
            with no_grad():
                x = Variable(np.random.randn(100000))
                y = square(square(square(x)))

    def test_memory_leak(self):
        memory_out = memory_usage(self.cause_memory_leak)
        self.assertTrue(memory_out[-1]/memory_out[0] < 2)
        # print(memory_out[0])
        # print(memory_out[-1])


if __name__ == '__main__':
    unittest.main()
