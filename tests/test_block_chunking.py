import unittest
from vegtamr.lic._parallel_by_block import _estimate_L1_cache_capacity, _generate_blocks


class TestBlockChunking(unittest.TestCase):

  def setUp(self):
    self.num_rows = 512
    self.num_cols = 512
    self.streamlength = 15

  def test_each_block_fits_in_L1_cache(self):
    max_cells_per_block_axis = _estimate_L1_cache_capacity(num_values_per_cell=1)
    L1_cache_cell_capacity   = max_cells_per_block_axis * max_cells_per_block_axis
    block_info               = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=True)
    data_ranges              = block_info["data_ranges"]
    for (row_lo, row_hi, col_lo, col_hi) in data_ranges:
      block_area = (row_hi - row_lo) * (col_hi - col_lo)
      self.assertLessEqual(
        block_area,
        L1_cache_cell_capacity,
        f"Block {((row_lo, row_hi), (col_lo, col_hi))} exceeds L1 cache capacity: {block_area} > {L1_cache_cell_capacity}"
      )

  def test_iter_range_covers_entire_grid(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    iter_ranges = blocks["iter_ranges"]
    covered_cells = set()
    for (row_lo, row_hi, col_lo, col_hi) in iter_ranges:
      for row_index in range(row_lo, row_hi):
        for col_index in range(col_lo, col_hi):
          covered_cells.add((row_index, col_index))
    expected_cells = set(
      (row_index, col_index)
      for row_index in range(self.num_rows)
      for col_index in range(self.num_cols)
    )
    uncovered_cells = expected_cells - covered_cells
    self.assertEqual(len(uncovered_cells), 0, f"Uncovered cells in iteration range: {len(uncovered_cells)}")

  def test_data_range_stays_within_bounds_when_not_periodic(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    data_ranges = blocks["data_ranges"]
    for (row_lo, row_hi, col_lo, col_hi) in data_ranges:
      self.assertGreaterEqual(row_lo, 0)
      self.assertLessEqual(row_hi, self.num_rows)
      self.assertGreaterEqual(col_lo, 0)
      self.assertLessEqual(col_hi, self.num_cols)

  def test_iter_range_is_subset_of_data_range(self):
    blocks = _generate_blocks(self.num_rows, self.num_cols, self.streamlength, use_periodic=False)
    iter_ranges = blocks["iter_ranges"]
    data_ranges = blocks["data_ranges"]
    for iter_block, data_block in zip(iter_ranges, data_ranges):
      (iter_row_lo, iter_row_hi, iter_col_lo, iter_col_hi) = iter_block
      (data_row_lo, data_row_hi, data_col_lo, data_col_hi) = data_block
      self.assertGreaterEqual(iter_row_lo, data_row_lo)
      self.assertLessEqual(iter_row_hi, data_row_hi)
      self.assertGreaterEqual(iter_col_lo, data_col_lo)
      self.assertLessEqual(iter_col_hi, data_col_hi)


if __name__ == "__main__":
  unittest.main()


## END OF TEST SCRIPT