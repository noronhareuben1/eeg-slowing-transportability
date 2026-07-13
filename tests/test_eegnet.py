import torch

from rcd.models.eegnet import ComplexityFusionNet, EEGNet


def test_eegnet_and_fusion_shapes() -> None:
    eegnet = EEGNet(n_channels=19, n_classes=3)
    values = torch.randn(2, 19, 1000)
    output = eegnet(values)
    assert output.logits.shape == (2, 3)
    assert output.embedding.shape == (2, eegnet.embedding_size)
    fusion = ComplexityFusionNet(eegnet, n_features=12, n_classes=3)
    fused = fusion(values, torch.randn(2, 12))
    assert fused.logits.shape == (2, 3)
