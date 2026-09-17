"""Built-in dataset specifications.

These replace per-experiment YAML so a caller can load a dataset the same way
torchvision loads MNIST: name, root, split, download.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

# Canonical modes used when writing window arrays. Aliases are accepted at load time.
FILLING_MODES = ("forward", "none", "zero")
FILLING_ALIASES = {
    "forward": "forward",
    "ffill": "forward",
    "none": "none",
    "nan": "none",
    "null": "none",
    "zero": "zero",
    "0": "zero",
}


def normalize_filling(filling: str) -> str:
    """Map user-facing filling names to a canonical mode."""
    key = str(filling).strip().lower()
    if key not in FILLING_ALIASES:
        allowed = ", ".join(FILLING_MODES) + " (aliases: nan, ffill, 0)"
        raise ValueError(f"Unknown filling '{filling}'. Choose one of: {allowed}.")
    return FILLING_ALIASES[key]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one vehicular cybersecurity dataset."""

    name: str
    family: str  # "can" or "v2x"
    description: str
    citation: str
    source_url: str
    downloadable: bool
    features: tuple[str, ...]
    attributes: tuple[str, ...]
    org_features: tuple[str, ...] = ()
    window_size: int = 50
    step_size: int = 10
    sampling_period: int = 1
    sampling_period_factors: tuple[int, ...] = ()
    filling: str = "forward"
    channels: int = 1
    label_index: int = 2
    classes: tuple[str, ...] = ("benign", "attack")
    status: str = "ready"  # ready | manual | planned
    notes: str = ""

    @property
    def num_signals(self) -> int:
        return len(self.features)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Channel-first window shape, matching a 1-channel image."""
        return (self.channels, self.window_size, self.num_signals)

    def with_overrides(
        self,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        sampling_period: Optional[int] = None,
        filling: Optional[str] = None,
    ) -> "DatasetSpec":
        updates = {}
        if window_size is not None:
            updates["window_size"] = int(window_size)
        if step_size is not None:
            updates["step_size"] = int(step_size)
        if sampling_period is not None:
            updates["sampling_period"] = int(sampling_period)
        if filling is not None:
            if self.family != "can":
                raise ValueError(
                    f"filling overrides are only valid for CAN datasets "
                    f"(got family={self.family!r} for {self.name})."
                )
            updates["filling"] = normalize_filling(filling)
        return replace(self, **updates) if updates else self


SYNCAN_FEATURES = (
    "ID_2_Sig_1", "ID_7_Sig_1", "ID_3_Sig_2", "ID_10_Sig_1", "ID_9_Sig_1",
    "ID_1_Sig_1", "ID_10_Sig_4", "ID_2_Sig_2", "ID_10_Sig_3", "ID_6_Sig_1",
    "ID_5_Sig_2", "ID_4_Sig_1", "ID_5_Sig_1", "ID_2_Sig_3", "ID_8_Sig_1",
    "ID_6_Sig_2", "ID_10_Sig_2", "ID_7_Sig_2", "ID_1_Sig_2", "ID_3_Sig_1",
)

# ID-2,6,9 -> 15/5, ID-4,10 -> 25/10, ID-1,3,5,7,8 -> 5/1
SYNCAN_PERIOD_FACTORS = (5, 1, 1, 10, 5, 1, 10, 5, 10, 5, 1, 10, 1, 5, 1, 5, 10, 1, 1, 1)

ROAD_FEATURES = (
    "ID_1413_Sig_7", "ID_930_Sig_5", "ID_1621_Sig_6", "ID_186_Sig_7", "ID_692_Sig_2",
    "ID_1628_Sig_4", "ID_1255_Sig_2", "ID_1668_Sig_5", "ID_1760_Sig_4", "ID_1760_Sig_3",
    "ID_208_Sig_6", "ID_1760_Sig_2", "ID_1760_Sig_1", "ID_526_Sig_2", "ID_1176_Sig_4",
    "ID_167_Sig_6", "ID_208_Sig_3", "ID_1455_Sig_14", "ID_661_Sig_1", "ID_192_Sig_1",
)

CAN_ORG_FEATURES = (
    "Label", "Time", "ID",
    "Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID",
)

ROAD_ORG_FEATURES = CAN_ORG_FEATURES + tuple(
    f"Signal{i}_of_ID" for i in range(5, 23)
)

CAN_ATTRIBUTES = ("Time", "ID", "Label", "File")

MISBEHAVIORX_FEATURES = (
    "speed_x", "del_pos_x", "speed_y", "del_pos_y",
    "accel_x", "del_speed_x", "accel_y", "del_speed_y",
    "del_heading_x", "yaw_rate_x", "del_heading_y", "yaw_rate_y",
)

MISBEHAVIORX_ATTRIBUTES = ("id", "time_chunk", "attack_gt", "attack_name")


SPECS: dict[str, DatasetSpec] = {
    "syncan": DatasetSpec(
        name="syncan",
        family="can",
        description="Synthetic CAN dataset with signal-level intrusion traces.",
        citation="Hanselmann et al., SynCAN, ETAS.",
        source_url="https://github.com/etas/SynCAN",
        downloadable=True,
        features=SYNCAN_FEATURES,
        attributes=CAN_ATTRIBUTES,
        org_features=CAN_ORG_FEATURES,
        window_size=50,
        step_size=10,
        sampling_period_factors=SYNCAN_PERIOD_FACTORS,
    ),
    "road": DatasetSpec(
        name="road",
        family="can",
        description="Real ORNL Automotive Dynamometer (ROAD) CAN intrusion dataset.",
        citation="Verma et al., ROAD, ORNL.",
        source_url="https://zenodo.org/records/10462796",
        downloadable=True,
        features=ROAD_FEATURES,
        attributes=CAN_ATTRIBUTES,
        org_features=ROAD_ORG_FEATURES,
        window_size=50,
        step_size=5,
        sampling_period_factors=tuple(1 for _ in ROAD_FEATURES),
    ),
    "misbehaviorx": DatasetSpec(
        name="misbehaviorx",
        family="v2x",
        description="VeReMi-extension / MisbehaviorX V2X misbehavior traces.",
        citation="Kamel et al., VeReMi Extension.",
        source_url="https://github.com/josephkamel/VeReMi-Dataset",
        downloadable=False,
        features=MISBEHAVIORX_FEATURES,
        attributes=MISBEHAVIORX_ATTRIBUTES,
        window_size=10,
        step_size=10,
        status="manual",
        notes=(
            "Place curated ambient/ and attacks/ folders under the dataset root. "
            "Automatic download is not wired yet."
        ),
    ),
    "x-canids": DatasetSpec(
        name="x-canids",
        family="can",
        description="X-CANIDS in-vehicle intrusion dataset.",
        citation="Jeong et al., X-CANIDS.",
        source_url="https://ieee-dataport.org/open-access/x-canids-dataset",
        downloadable=False,
        features=(),
        attributes=CAN_ATTRIBUTES,
        status="planned",
        notes="Registered so experiments can target it, but the loader is not implemented yet.",
    ),
}

ALIASES = {
    "vasp": "misbehaviorx",
    "veremi": "misbehaviorx",
    "syncan_robids": "syncan",
}


def get_spec(name: str) -> DatasetSpec:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in SPECS:
        known = ", ".join(sorted(SPECS))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}")
    return SPECS[key]


def list_specs(status: Optional[str] = None) -> list[DatasetSpec]:
    specs = list(SPECS.values())
    if status is not None:
        specs = [spec for spec in specs if spec.status == status]
    return specs
