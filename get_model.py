from vanilla_models import VanillaResNet50, VanillaResNet18, VanillaAlexNet
from resnet_gat import gat_resnet18, gat_resnet50, gat_resnet101
from resnet_se import se_resnet18, se_resnet50
from resnet_eca import eca_resnet18, eca_resnet50
from resnet_cbam import cbam_resnet18, cbam_resnet50
from resnet_sa import sa_resnet18, sa_resnet50
# from conv2former import Conv2former
from nlnet import nln_resnet50

def get_model(model_name, model_type, num_classes, args):
    eca_k = {
        "resnet18": [3, 5, 5, 5],
        #  "resnet50": [64, 16, 4, 1]
    }

    model = None
    if model_name == "resnet50":
        if model_type == "gnn":
            model = gat_resnet50(num_classes=num_classes, args=args)
        elif model_type == "se":
            model = se_resnet50(num_classes=num_classes)
        elif model_type == "eca":
            model = eca_resnet50(eca_k["resnet50"], num_classes=num_classes)
        elif model_type == "cbam":
            model = cbam_resnet50(num_classes=num_classes)
        elif model_type == "sa":
            model = sa_resnet50(num_classes=num_classes)
        elif model_type == "vanilla":
            model = VanillaResNet50(num_classes=num_classes)

    elif model_name == "resnet18":
        if model_type == "gnn":
            model = gat_resnet18(num_classes=num_classes, args=args)
        elif model_type == "se":
            model = se_resnet18(num_classes=num_classes)
        elif model_type == "eca":
            model = eca_resnet18(eca_k["resnet18"], num_classes=num_classes)
        elif model_type == "cbam":
            model = cbam_resnet18(num_classes=num_classes)
        elif model_type == "sa":
            model = sa_resnet18(num_classes=num_classes)
        elif model_type == "vanilla":
            model = VanillaResNet18(num_classes=num_classes)

    elif model_name == "alexnet":
        model = VanillaAlexNet(num_classes=num_classes)

    elif model_name == "resnet101":
        if model_type == "gnn":
            model = gat_resnet101(num_classes=num_classes, args=args)

    elif model_name == "conv2former":
        model = Conv2former()
    elif model_name == "nln":
        model = nln_resnet50()

    for param in model.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)#, param.data

    return model

