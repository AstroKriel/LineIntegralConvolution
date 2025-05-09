import unittest
from vegtamr.lic._parallel_by_block import _estimate_L1_cache_capacity, _generate_blocks


class TestBlockChunking(unittest.TestCase):

  def setUp(self):
    self.num_rows = 512
    self.num_cols = 512
    self.streamlength = 15

  def test_each_block_fits_in_L1_cache(self):
    # Get max block size per axis from cache estimate
    max_cells_per_block_axis = _estimate_L1_cache_capacity(num_values_per_cell=1)
    max_total_cells_in_L1 = max_cells_per_block_axis ** 2
    # Generate blocks
    block_info = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=True)
    data_ranges = block_info["data_ranges"]
    # Check each data block
    for (r0, r1, c0, c1) in data_ranges:
      block_area = (r1 - r0) * (c1 - c0)
      self.assertLessEqual(
        block_area,
        max_total_cells_in_L1,
        f"Block {((r0, r1), (c0, c1))} exceeds L1 cache capacity: {block_area} > {max_total_cells_in_L1}"
      )

  def test_iter_range_covers_entire_grid(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    iter_ranges = blocks["iter_ranges"]
    covered_cells = set()
    for r0, r1, c0, c1 in iter_ranges:
      for r in range(r0, r1):
        for c in range(c0, c1):
          covered_cells.add((r, c))
    expected_cells = set((r, c) for r in range(self.num_rows) for c in range(self.num_cols))
    uncovered_cells = expected_cells - covered_cells
    self.assertEqual(len(uncovered_cells), 0, f"Uncovered cells in iteration range: {len(uncovered_cells)}")

  def test_data_range_stays_within_bounds_when_not_periodic(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    data_ranges = blocks["data_ranges"]
    for r0, r1, c0, c1 in data_ranges:
      self.assertGreaterEqual(r0, 0)
      self.assertLessEqual(r1, self.num_rows)
      self.assertGreaterEqual(c0, 0)
      self.assertLessEqual(c1, self.num_cols)

  def test_iter_range_is_subset_of_data_range(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    iter_ranges = blocks["iter_ranges"]
    data_ranges = blocks["data_ranges"]
    for iter_blk, data_blk in zip(iter_ranges, data_ranges):
      r0_i, r1_i, c0_i, c1_i = iter_blk
      r0_d, r1_d, c0_d, c1_d = data_blk
      self.assertGreaterEqual(r0_i, r0_d)
      self.assertLessEqual(r1_i, r1_d)
      self.assertGreaterEqual(c0_i, c0_d)
      self.assertLessEqual(c1_i, c1_d)


if __name__ == "__main__":
  unittest.main()


## END OF TEST SCRIPT