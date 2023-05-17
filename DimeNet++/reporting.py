from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    maxerr = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return([mae, rmse, maxerr, r2])
