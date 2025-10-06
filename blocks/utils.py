
import matplotlib.pyplot as plt

def plot_tensors_2d(q_i, k_i, dot_product):
  q_np = q_i.numpy()
  k_np = k_i.numpy()

  fig = plt.figure(figsize=(6, 2.5))
  ax = fig.add_subplot(111)

  ax.quiver(0, 0, q_np[0], q_np[1] ,
            angles='xy', scale_units='xy', scale=1, 
            color='blue', label='q_i', linewidth=3)

  ax.quiver(0, 0, k_np[0], k_np[1],
            angles='xy', scale_units='xy', scale=1, 
            color='red', label='k_i', linewidth=3)

  ax.text(0.05, 0.95, f'Produto escalar: {dot_product.item():.4f}',
          transform=ax.transAxes, fontsize=12,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.legend()

  ax.grid(True)

  plt.show()