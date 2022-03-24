from keras import losses


def custom_loss(y_true, y_pred):
    # return cosine_distance + MAE
    cosine = losses.cosine_proximity(y_true, y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    loss = (cosine) + mle
    return loss


def custom_loss_2(y_true, y_pred):
    # scaled cosine_distance with MES + MAE
    cosine = losses.cosine_proximity(y_true, y_pred)
    mse = losses.mean_squared_error(y_true, y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    loss = (1 + cosine) * mse + mle
    return loss
