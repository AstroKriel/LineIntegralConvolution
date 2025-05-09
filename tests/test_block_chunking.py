import unittest
from vegtamr.lic._parallel_by_block import _estimate_L1_cache_capacity, _generate_blocks


class TestBlockChunking(unittest.TestCase):

  def test_each_block_fits_in_L1_cache(self):
    # Domain size and LIC streamlength
    num_rows = 512
    num_cols = 512
    streamlength = 15
    # Get max block size per axis from cache estimate
    max_cells_per_block_axis = _estimate_L1_cache_capacity(num_values_per_cell=1)
    max_total_cells_in_L1 = max_cells_per_block_axis ** 2
    # Generate blocks
    block_info = _generate_blocks(num_rows, num_cols, streamlength, use_periodic=True)
    data_ranges = block_info["data_ranges"]
    # Check each data block
    for (r0, r1, c0, c1) in data_ranges:
      block_area = (r1 - r0) * (c1 - c0)
      self.assertLessEqual(
        block_area,
        max_total_cells_in_L1,
        f"Block {((r0, r1), (c0, c1))} exceeds L1 cache capacity: {block_area} > {max_total_cells_in_L1}"
      )


if __name__ == "__main__":
  unittest.main()


## END OF TEST SCRIPT