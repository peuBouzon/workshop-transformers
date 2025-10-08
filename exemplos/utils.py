from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
import plotly.express as px
from sklearn.metrics import precision_recall_curve
import torch

def get_class_weights(num_classes, df, y_train, int_to_label, device):
    if num_classes > 2:
        weights = 1 / (torch.bincount(torch.tensor(y_train)) / len(y_train)).to(device)
        print(f"Class Weights:")
        for i, weight in enumerate(weights):
            print(f"  - Class '{int_to_label[i]}': {weight:.2f}")
    else:
        weights = torch.tensor([df['label'].value_counts()[0] / df['label'].value_counts()[1]]).to(device)
        print(weights)

def plot_sample_images(df, IMAGE_DIR):
  unique_df = df.groupby('diagnostic').sample(n=1, random_state=16).reset_index(drop=True)
  unique_df['diagnostic'] = pd.Categorical(unique_df['diagnostic'], categories=['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK'], ordered=True)
  unique_df = unique_df.sort_values('diagnostic').reset_index(drop=True)

  n_rows = 2
  n_cols = 3

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
  fig.patch.set_facecolor('white')

  axes = axes.flatten()

  for i, ax in enumerate(axes):
      ax.axis('off')
      if i < len(unique_df):
          img_path = Path(IMAGE_DIR) / unique_df['img_id'][i]
          diagnosis_names = {
              'BCC': 'Carcinoma Basocelular',
              'SCC': 'Carcinoma Espinocelular',
              'NEV': 'Nevo',
              'MEL': 'Melanoma',
              'ACK': 'Queratose Actínica',
              'SEK': 'Queratose Seborreica'
          }
          current_diagnosis = diagnosis_names[unique_df['diagnostic'][i]]

          img = mpimg.imread(img_path)
          ax.imshow(img)
          ax.set_title(current_diagnosis, fontsize=10, fontweight='bold')

  fig.suptitle(
      'Imagens por Diagnóstico',
      fontsize=20,
      fontweight='bold',
      y=0.98
  )
  plt.tight_layout(rect=[0, 0, 1, 1])
  plt.show()


def plot_precision_recall_curve(preds, labels, min_recall=0.8):
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)

    target_idxs = np.where(recall >= min_recall)[0]
    if len(target_idxs) > 0:
        target_idx = target_idxs[np.argmax(precision[target_idxs])]
        target_threshold = thresholds[target_idx]
        target_precision = precision[target_idx]
        target_recall_actual = recall[target_idx]

    plt.figure(figsize=(5, 3))
    plt.annotate(
        f'Precision = {target_precision:.2f}\nRecall = {target_recall_actual:.2f}\nThreshold = {target_threshold:.5f}',
        xy=(target_recall_actual, target_precision),
        xytext=(target_recall_actual + 0.05, target_precision - 0.15),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.4)
    )

    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Curva PR (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=15)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

def plot_precision_recall_iterative(y_test, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    pr_df = pd.DataFrame({
        'recall': recall[:-1],
        'precision': precision[:-1],
        'threshold': thresholds
    })

    fig = px.line(
        pr_df,
        x='recall',
        y='precision',
        title='Curva Precision-Recall',
        labels={'recall': 'Recall', 'precision': 'Precision'},
        hover_data=['threshold']
    )

    fig.update_traces(hovertemplate=(
        '<b>Recall</b>: %{x:.3f}<br>'
        '<b>Precision</b>: %{y:.3f}<br>'
        '<b>Threshold</b>: %{customdata[0]:.3f}'
        '<extra></extra>'
    ))

    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig.add_shape(
        type='line',
        x0=0, y0=no_skill, x1=1, y1=no_skill,
        line=dict(color='RoyalBlue', width=2, dash='dash'),
        name='No Skill'
    )

    fig.update_layout(
        xaxis_range=[0, 1.01],
        yaxis_range=[0, 1.05],
        legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom")
    )

    fig.show()

def plot_interactive_roc_curve(y_true, y_probs, title='Curva ROC'):
    # 1. Calculate the ROC curve values
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # 2. Calculate the Area Under the Curve (AUC)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # 3. Create a DataFrame for Plotly
    # The thresholds array is usually one element longer than needed, so we trim it.
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        # We add a placeholder for the first threshold, as roc_curve doesn't provide it
        'thresholds': np.concatenate(([1.0], thresholds[1:])) 
    })
    
    # 4. Create the interactive plot
    fig = px.line(
        roc_df,
        x='fpr',
        y='tpr',
        hover_data=['thresholds']
    )
    
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='RoyalBlue'),
        x0=0, y0=0, x1=1, y1=1
    )
    
    fig.update_traces(hovertemplate=(
        '<b>FPR</b>: %{x:.3f}<br>'
        '<b>TPR (Recall)</b>: %{y:.3f}<br>'
        '<b>Threshold</b>: %{customdata[0]:.3f}'
        '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'<b>{title} (AUC = {roc_auc:.3f})</b>',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR / Recall)',
        xaxis=dict(range=[0, 1.01]),
        yaxis=dict(range=[0, 1.01]),
        showlegend=False
    )
    
    fig.show()
