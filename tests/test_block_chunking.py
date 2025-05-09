import unittest
import numpy as np
from vegtamr.lic._parallel_by_block import _calculate_block_size, _split_into_blocks

class TestBlockChunking(unittest.TestCase):
  def setUp(self):
    self.streamlength = 20
    self.cache_size = 8 * 1024**2  # 8MB

  def test_block_size_calculation(self):
    # Test valid cases
    with self.subTest("Medium array"):
      self.assertEqual(_calculate_block_size((512, 512), 20), 884)
    
    with self.subTest("Small array"):
      self.assertIsNone(_calculate_block_size((50, 50), 15))
    
    with self.subTest("Streamlength too large"):
      self.assertIsNone(_calculate_block_size((100, 100), 30))

  def test_block_splitting(self):
    # Test block generation logic
    blocks = _split_into_blocks((256, 256), 20)
    
    with self.subTest("Block count"):
      self.assertEqual(len(blocks), 16)  # (256/64)^2
      
    with self.subTest("Block coverage"):
      covered = np.zeros((256, 256), bool)
      for r, c, bs in blocks:
        covered[r:r+bs, c:c+bs] = True
      self.assertTrue(covered.all())

    with self.subTest("Edge handling"):
      blocks = _split_into_blocks((100, 100), 20)
      last_block = blocks[-1]
      self.assertEqual(last_block[0] + last_block[2], 100)

class TestBlockProcessing(unittest.TestCase):
  def test_block_padding(self):
    from vegtamr.lic._parallel_by_block import _process_block
    # Test padding extraction logic
    vfield = np.random.rand(2, 100, 100).astype(np.float32)
    sfield = np.random.rand(100, 100).astype(np.float32)
    
    # Mock shared memory
    class MockShm:
      def __init__(self, arr):
        self.arr = arr
      def buf(self):
        return self.arr
    
    results = _process_block(
      32, 32, 64,
      None, vfield.shape, vfield.dtype,
      None, sfield.shape, sfield.dtype,
      20, True
    )
    
    with self.subTest("Result keys"):
      self.assertEqual(len(results), 64*64)
      
    with self.subTest("Normalization"):
      self.assertTrue(0 <= results[(64,64)] <= 1.0)

if __name__ == '__main__':
  unittest.main()
