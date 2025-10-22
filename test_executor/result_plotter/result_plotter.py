import matplotlib.pyplot as plt
import numpy as np


class ResultPlotter:
    def __init__(self):
        pass

    def plot(self, results):
        sizes = [result['size'] for result in results]
        relative_accuracy = [result['relative_factorization_accuracy'] for result in results]
        residuals = [result['residual'] for result in results]
        growth_factors = [result['growth_factor'] for result in results]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        linestyles = ['-', '--', '-.']
        strategies = ['No Pivoting', 'Partial Pivoting', 'Full Pivoting']
        
        # Plot relative factorization accuracy.
        for i in range(3):
            y_values = [acc[i] for acc in relative_accuracy]
            ax1.semilogy(sizes, y_values, color=colors[i], marker=markers[i], 
                        linestyle=linestyles[i], label=strategies[i], linewidth=2, markersize=6,
                        alpha=0.8)
        ax1.set_ylabel('Relative Factorization Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot residual.
        for i in range(3):
            y_values = [res[i] for res in residuals]
            ax2.semilogy(sizes, y_values, color=colors[i], marker=markers[i], 
                        linestyle=linestyles[i], label=strategies[i], linewidth=2, markersize=6,
                        alpha=0.8)
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot growth factor.
        for i in range(3):
            y_values = [gf[i] for gf in growth_factors]
            ax3.semilogy(sizes, y_values, color=colors[i], marker=markers[i], 
                        linestyle=linestyles[i], label=strategies[i], linewidth=2, markersize=6,
                        alpha=0.8)
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Growth Factor')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
