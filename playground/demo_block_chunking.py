from collections import Counter
import matplotlib.pyplot as mpl_plt
import matplotlib.patches as mpl_patches
from vegtamr.lic._parallel_by_block import _generate_blocks


def print_block_shape_stats(block_type, block_ranges):
  block_shapes = [
    (
      row_stop - row_start,
      col_stop - col_start
    )
    for (row_start, row_stop, col_start, col_stop) in block_ranges
  ]
  shape_counts  = Counter(block_shapes)
  sorted_shapes = sorted(
    shape_counts.items(),
    key = lambda x: (
      x[0][0] * x[0][1],
      x[0]
    )
  )
  print(f"count of `{block_type}` block shapes:")
  for (rows, cols), count in sorted_shapes:
    print(f"\t- {rows}x{cols}: {count}")


def plot_blocking(ax, block_ranges, *, color, lw=1, use_fill=False):
  for (row_start, row_stop, col_start, col_stop) in block_ranges:
    block = mpl_patches.Rectangle(
      (col_start, row_start),
      col_stop - col_start,
      row_stop - row_start,
      linewidth = lw,
      edgecolor = color,
      facecolor = color if use_fill else "none",
      alpha     = 0.5
    )
    ax.add_patch(block)


def main():
  num_cells    = 325
  streamlength = 10
  ## generate blocks
  block_info  = _generate_blocks(num_rows=num_cells, num_cols=num_cells, streamlength=streamlength)
  iter_ranges = block_info["iter_ranges"]
  data_ranges = block_info["data_ranges"]
  ## print info
  print(f"domain size: {num_cells}x{num_cells}")
  print(f"streamlength: {streamlength}")
  print(" ")
  print(f"generated {len(iter_ranges)} cache-aware blocks.")
  print_block_shape_stats("iter_ranges", iter_ranges)
  print_block_shape_stats("data_ranges", data_ranges)
  print(" ")
  ## generate figure
  fig, ax = mpl_plt.subplots(figsize=(10, 10))
  plot_padding = 0.1 * num_cells
  ax.set_xlim(-plot_padding, num_cells + plot_padding)
  ax.set_ylim(-plot_padding, num_cells + plot_padding)
  ax.set_aspect("equal")
  ax.invert_yaxis()
  ax.set_xlabel("columns")
  ax.set_ylabel("rows")
  ax.grid(True, color="black", linestyle="--", alpha=0.5)
  ## plot blocks
  plot_blocking(ax, [(0, num_cells, 0, num_cells)], color="black", lw=2)
  plot_blocking(ax, iter_ranges, color="cornflowerblue")
  plot_blocking(ax, [data_ranges[6]], color="orangered", use_fill=True)
  ## show save
  fig.savefig("block_debug_plot.png", dpi=150)
  mpl_plt.show()


if __name__ == "__main__":
  main()
