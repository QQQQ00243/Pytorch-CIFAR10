from torch import nn


def get_fc(
    in_features,
    width_hiddens,
    num_classes,
    act_name="ReLU",
    flatten=True,
    reg_name="None",
):
    assert reg_name in ["bn", "dropout_0.25", "dropout_0.5", "None"]
    layers = []
    if flatten:
        layers.append(nn.Flatten())
    act = getattr(nn, act_name)()
    for out_features in width_hiddens:
        layers.append(nn.Linear(in_features, out_features))
        if reg_name == "None":
            pass
        elif reg_name == "bn":
            layers.append(nn.BatchNorm1d(out_features))
        else:
            dropout_rate = float(reg_name.split("_")[-1])
            layers.append(nn.Dropout(dropout_rate))
        layers.append(act)
        in_features = out_features
    layers.append(nn.Linear(in_features, num_classes))
    model = nn.Sequential(*layers)
    return model
