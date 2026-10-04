import argparse

import pytest

from eldercare.run import parse_args, parse_bed_roi, validate_args


def test_manual_bed_roi_validation():
  box = parse_bed_roi("10,20,110,220")
  assert box.to_list() == [10.0, 20.0, 110.0, 220.0]

  with pytest.raises(argparse.ArgumentTypeError):
    parse_bed_roi("10,20,5,220")
  with pytest.raises(argparse.ArgumentTypeError):
    parse_bed_roi("10,20,110")


def test_command_line_requires_valid_input_and_rates():
  with pytest.raises(SystemExit):
    validate_args(parse_args([]))
  with pytest.raises(SystemExit):
    validate_args(parse_args(["--video", "room.mp4", "--analysis-fps", "0"]))

  args = parse_args(["--video", "room.mp4", "--analysis-fps", "4"])
  validate_args(args)
