# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pypylon import pylon

from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_basler"]


def discover_basler() -> list[DeviceInfo]:
    factory = pylon.TlFactory.GetInstance()
    results: list[DeviceInfo] = []

    for i, dev in enumerate(factory.EnumerateDevices()):
        results.append(
            DeviceInfo(
                device_id=dev.GetSerialNumber(),
                index=i,
                name=dev.GetUserDefinedName(),
                driver="basler",
                hardware_id=dev.GetSerialNumber(),
                manufacturer=dev.GetVendorName(),
                model=dev.GetModelName(),
                metadata={"address": dev.GetAddress()},
            )
        )

    return results
