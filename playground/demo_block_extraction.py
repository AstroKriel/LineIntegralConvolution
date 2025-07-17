import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vegtamr.utils import vfields
from vegtamr.lic._parallel_by_block import _generate_blocks, _extract_blocks


def plot_block(ax, vblock, sblock, title, color="black", step=2):
  # ax.imshow(sblock, cmap="gray", origin="upper")s
  Y, X = np.mgrid[0:vblock.shape[1]:step, 0:vblock.shape[2]:step]
  U = vblock[1, ::step, ::step]
  V = vblock[0, ::step, ::step]
  magnitude = np.sqrt(U**2 + V**2) + 1e-8
  U_norm = U / magnitude
  V_norm = V / magnitude
  ax.quiver(X, Y, U_norm, -V_norm, color=color, scale=50, pivot="middle")
  ax.set_title(title)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])


def main():
  print("Loading vector field...")
  vfield_dict  = vfields.vfield_lotka_volterra(100)
  vfield       = vfield_dict["vfield"]
  streamlength = vfield_dict["streamlength"]
  name         = vfield_dict["name"]
  print("Generating synthetic scalar field...")
  sfield = np.random.rand(*vfield.shape[1:])
  print("Generating blocks...")
  blocks        = _generate_blocks(vfield.shape[1], vfield.shape[2], streamlength)
  data_ranges   = blocks["data_ranges"]
  num_blocks    = len(data_ranges)
  block_indices = list(range(min(3, num_blocks)))
  extracted     = _extract_blocks([data_ranges[block_index] for block_index in block_indices], vfield, sfield)
  print("Plotting full domain + extracted blocks...")
  fig, axs = plt.subplots(
    1,
    1 + len(block_indices),
    figsize=(
      5 * (1 + len(block_indices)),
      5
    )
  )
  if len(block_indices) == 1: axs = [axs]
  plot_block(axs[0], vfield, sfield, title="full domain", color="red")
  for ax, block_index, vblock, sblock in zip(
      axs[1:],
      block_indices,
      extracted["vfield_blocks"],
      extracted["sfield_blocks"]
    ):
    plot_block(ax, vblock, sblock, title=f"block {block_index}", color="blue")
  plt.tight_layout()
  output_path = Path(__file__).resolve().parent / f"extracted_blocks_with_domain_{name}.png"
  plt.savefig(output_path, dpi=200)
  plt.show()
  print("Saved to:", output_path)


if __name__ == "__main__":
  main()
