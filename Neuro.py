#!/usr/bin/env python3
"""Точка входа NeuroLSTM 2.0.

Запуск: python Neuro.py --help
Математические реализации находятся в пакете neurolstm; для обновления
репозитория копируется весь архив, а не только этот небольшой файл.
Старые checkpoint .pth намеренно не выдаются за веса новой архитектуры.
"""
from neurolstm.model import AdaptiveLSTM, AdaptiveLSTMCell
from neurolstm.physics import gas_inventory, gas_mass_residual, gas_loss
from neurolstm.gradients import combine_gradients
from neurolstm.cli import main

__all__ = ["AdaptiveLSTM", "AdaptiveLSTMCell", "gas_inventory", "gas_mass_residual",
           "gas_loss", "combine_gradients", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
