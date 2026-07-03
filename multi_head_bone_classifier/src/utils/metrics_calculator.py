from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics import Metric
from torchmetrics.wrappers import ClasswiseWrapper
import json


class MetricsCalculator(Metric):
    def __init__(
        self, num_classes1, num_classes2, body_part_names_path, view_names_path
    ):
        super().__init__()

        # body_part_names_path = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/body_part_names.json"
        # view_names_path = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/view_names.json"

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.body_part_names_path = body_part_names_path
        self.view_names_path = view_names_path

        with open(self.body_part_names_path, "r") as file:
            self.body_part_names = json.load(file)
        with open(self.view_names_path, "r") as file:
            self.view_names = json.load(file)

        self.body_metrics = MetricCollection(
            MulticlassAccuracy(num_classes=num_classes1, average="macro"),
            MulticlassF1Score(num_classes=num_classes1, average="macro"),
            prefix="body_",
        )
        self.body_metrics_micro = MetricCollection(
            MulticlassAccuracy(num_classes=num_classes1, average="micro"),
            MulticlassF1Score(num_classes=num_classes1, average="micro"),
            prefix="body_micro",
        )
        self.view_metrics = MetricCollection(
            MulticlassAccuracy(num_classes=num_classes2, average="macro"),
            MulticlassF1Score(num_classes=num_classes2, average="macro"),
            prefix="view_",
        )

        self.body_classwise = ClasswiseWrapper(
            MulticlassAccuracy(num_classes=num_classes1, average=None),
            labels=self.body_part_names,
            prefix="val_body_classwise/",
        )

        self.view_classwise = ClasswiseWrapper(
            MulticlassAccuracy(num_classes=num_classes2, average=None),
            labels=self.view_names,
            prefix="val_view_classwise/",
        )

    def update(self):
        pass

    def compute(self):
        pass

    def get_dict(self, y_hat1, y1, y_hat2, y2, losses, prefix):
        # Compute metrics
        body_metrics = self.body_metrics(y_hat1, y1)
        view_metrics = self.view_metrics(y_hat2, y2)
        body_metrics_micro = self.body_metrics_micro(y_hat1, y1)

        # Append them to losses dictionary
        losses.update(body_metrics)
        losses.update(view_metrics)
        losses.update(body_metrics_micro)

        # Add prefix to all keys
        log = {prefix + str(key): val for key, val in losses.items()}
        log[prefix + "summed_f1"] = (
            log[prefix + "view_MulticlassF1Score"]
            + log[prefix + "body_MulticlassF1Score"]
        )
        return log

    def get_class_wise_dict(self, y_hat1, y1, y_hat2, y2):
        # Compute metrics
        body_metrics = self.body_classwise(y_hat1, y1)
        view_metrics = self.view_classwise(y_hat2, y2)

        # Append them into one dictionary
        body_metrics.update(view_metrics)
        return body_metrics
