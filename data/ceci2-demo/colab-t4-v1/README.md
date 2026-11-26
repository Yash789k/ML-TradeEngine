# Colab GPU training

The Tesla T4 notebook fitted twelve soft gates, six Q models that select an expert and two annual joint mixtures with newly initialized expert networks. The datasets, trading rules and annual splits match the local controller studies. The GPU uses its own minibatch order and numerical kernels. See each run's receipt for hardware, cutoffs and forecast comparisons. The daily ledgers, predictions and checkpoints were downloaded from the completed notebook.

[Notebook with outputs](https://colab.research.google.com/drive/1qjCtnMsCAxeps4P7_RBPiJtPpmHHSqQD?authuser=1) · [Soft gates](custom-gates/receipt.json) · [Expert selection](expert-selector/receipt.json) · [Joint training](joint-experts/result.json)
