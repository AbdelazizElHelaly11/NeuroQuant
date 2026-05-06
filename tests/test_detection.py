"""Object-detection model loader tests.

Covers the detection branch of ``ModelLoader`` introduced alongside
``QuantizationConfig.task``: Faster R-CNN head adaptation, the warning
path for non-RCNN architectures (RetinaNet/SSD/FCOS), and the
unknown-name error path.
"""

from __future__ import annotations

import logging

import pytest

from config import QuantizationConfig
from models.model_loader import ModelLoader


@pytest.fixture(scope="module")
def detection_cfg_factory():
    """Factory producing a ready-to-load detection config."""

    def _make(model_name: str, num_classes: int = 5):
        cfg = QuantizationConfig(
            model_name=model_name,
            num_classes=num_classes,
            task="detection",
            input_shape=(3, 300, 300),
        )
        return cfg

    return _make


def test_faster_rcnn_box_predictor_adapted(detection_cfg_factory):
    cfg = detection_cfg_factory("fasterrcnn_resnet50_fpn", num_classes=5)
    model = ModelLoader(cfg).load()
    out_features = model.roi_heads.box_predictor.cls_score.out_features
    assert out_features == cfg.num_classes


def test_retinanet_loads_with_warning(detection_cfg_factory, caplog):
    cfg = detection_cfg_factory("retinanet_resnet50_fpn", num_classes=7)
    with caplog.at_level(logging.WARNING, logger="neuroquant"):
        ModelLoader(cfg).load()
    assert any(
        "automatic head adaptation is not supported" in rec.message.lower()
        for rec in caplog.records
    )


def test_unknown_detection_name_raises(detection_cfg_factory):
    cfg = detection_cfg_factory("not_a_real_detection_model")
    with pytest.raises(ValueError, match="Unknown detection model"):
        ModelLoader(cfg).load()


def test_classification_path_unaffected():
    cfg = QuantizationConfig(
        model_name="mobilenetv2",
        num_classes=5,
        task="classification",
        input_shape=(3, 32, 32),
    )
    model = ModelLoader(cfg).load()
    last_linear = next(
        m for m in reversed(list(model.modules()))
        if m.__class__.__name__ == "Linear"
    )
    assert last_linear.out_features == cfg.num_classes
