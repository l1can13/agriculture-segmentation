import segmentation_models_pytorch as smp

MODEL_SPECS = {
    "unet_resnet50": {
        "title": "U-Net + ResNet-50",
        "arch": "Unet",
        "encoder": "resnet50",
        "kwargs": {},
    },
    "upernet_convnext_tiny": {
        "title": "UPerNet + ConvNeXt-Tiny",
        "arch": "UPerNet",
        "encoder": "tu-convnext_tiny.fb_in1k",
        "kwargs": {},
    },
    "dpt_dinov3_vits16": {
        "title": "DPT + DINOv3 ViT-S/16",
        "arch": "DPT",
        "encoder": "tu-vit_small_patch16_dinov3.lvd1689m",
        "kwargs": {
            "dynamic_img_size": True,
            "decoder_intermediate_channels": (48, 96, 192, 384),
            "decoder_fusion_channels": 64,
        },
    },
}


def build_model(name, num_classes, in_channels=4, pretrained=True):
    spec = MODEL_SPECS[name]
    model_cls = getattr(smp, spec["arch"])
    return model_cls(
        encoder_name=spec["encoder"],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        **spec["kwargs"],
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())
