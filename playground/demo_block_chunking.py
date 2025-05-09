import matplotlib.pyplot as mpl_plt
import matplotlib.patches as mpl_patches
from vegtamr.lic._parallel_by_block import _generate_blocks


def plot_blocking(ax, block_ranges, *, color, lw=1, use_fill=False):
  for (row_start, row_end, col_start, col_end) in block_ranges:
    rect = mpl_patches.Rectangle(
      (col_start, row_start),
      col_end - col_start,
      row_end - row_start,
      linewidth = lw,
      edgecolor = color,
      facecolor = color if use_fill else "none",
      alpha     = 0.5
    )
    ax.add_patch(rect)


def main():
  length       = 325
  streamlength = 10

  fig, ax = mpl_plt.subplots(figsize=(10, 10))
  plot_bump = 0.1 * length
  ax.set_xlim(-plot_bump, length + plot_bump)
  ax.set_ylim(-plot_bump, length + plot_bump)
  ax.set_aspect("equal")
  ax.invert_yaxis()
  ax.set_title(f"Cache-Aware Blocks\nDomain: {length}x{length}, Streamlength: {streamlength}")
  ax.set_xlabel("columns")
  ax.set_ylabel("rows")
  ax.grid(True, color="black", linestyle="--", alpha=0.5)

  block_info  = _generate_blocks(num_rows=length, num_cols=length, streamlength=streamlength)
  iter_ranges = block_info["iter_ranges"]
  data_ranges = block_info["data_ranges"]
  iter_cells_per_block_axis = block_info["iter_cells_per_block_axis"]
  tot_cells_per_block_axis  = block_info["tot_cells_per_block_axis"]

  plot_blocking(ax, [(0, length, 0, length)], color="black", lw=2)
  plot_blocking(ax, iter_ranges, color="cornflowerblue")
  plot_blocking(ax, [data_ranges[6]], color="orangered", use_fill=True)

  print(" ")
  print(f"Generated {len(iter_ranges)} iteration blocks.")
  print(f"Iteration block size: {iter_cells_per_block_axis}x{iter_cells_per_block_axis}")
  print(f"Total block size (with halo): {tot_cells_per_block_axis}x{tot_cells_per_block_axis}")
  print(" ")

  fig.savefig("block_debug_plot.png", dpi=150)
  mpl_plt.show()


if __name__ == "__main__":
  main()
