from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import joblib

# Polynomial Linear Regression
linear_model_poly = LinearRegression()
linear_model_poly.fit(X_train_poly, y_train)
y_pred_lr_poly = linear_model_poly.predict(X_test_poly)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# LSTM Model
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    Input(shape=(1, X_train_lstm.shape[2])),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32)
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Support Vector Machine (SVM)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Save Models
joblib.dump(rf_model, 'random_forest_weather_model.pkl')
lstm_model.save('lstm_weather_model.keras')
