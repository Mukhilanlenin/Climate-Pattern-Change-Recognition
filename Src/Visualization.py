import matplotlib.pyplot as plt
import seaborn as sns

# Bimodal distribution plots
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, color="#5DADEC", label='Actual Temperatures', fill=True, alpha=0.3)
sns.kdeplot(y_pred_lr_poly, color="#34A853", label='Polynomial Linear Regression', fill=True, alpha=0.3)
plt.title('Polynomial Linear Regression - Bimodal Distribution')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Random Forest
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, color="#5DADEC", label='Actual Temperatures', fill=True, alpha=0.3)
sns.kdeplot(y_pred_rf, color="#FF6F00", label='Random Forest Predictions', fill=True, alpha=0.3)
plt.title('Random Forest - Bimodal Distribution')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Multi-bar plot comparing models
plt.figure(figsize=(16, 8))
indices = np.arange(30)  # Limit to first 30 points
bar_width = 0.15

plt.bar(indices, y_test[:30], width=bar_width, color="#5DADEC", label='Actual Temperatures')
plt.bar(indices + bar_width, y_pred_lr_poly[:30], width=bar_width, color="#34A853", label='Linear Regression')
plt.bar(indices + 2 * bar_width, y_pred_rf[:30], width=bar_width, color="#FF6F00", label='Random Forest')
plt.bar(indices + 3 * bar_width, y_pred_lstm.flatten()[:30], width=bar_width, color="#E63946", label='LSTM')
plt.bar(indices + 4 * bar_width, y_pred_svm[:30], width=bar_width, color="#800080", label='SVM')

plt.xlabel('Test Data Points')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperatures by Models')
plt.legend()
plt.xticks(indices + 2 * bar_width, indices)
plt.tight_layout()
plt.show()
